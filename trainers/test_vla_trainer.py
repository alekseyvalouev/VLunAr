import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainers.vla_trainer import train_vla, VLADataset, collate_fn

class TestVLATrainer(unittest.TestCase):
    @patch('trainers.vla_trainer.VLA')
    @patch('trainers.vla_trainer.PaliGemmaForConditionalGeneration')
    @patch('trainers.vla_trainer.prepare_model_for_kbit_training')
    @patch('trainers.vla_trainer.get_peft_model')
    @patch('torch.save')
    def test_train_vla_flow(self, mock_save, mock_get_peft, mock_prepare, mock_pg, mock_vla_cls):
        # Setup mocks
        mock_vla_instance = MagicMock()
        mock_vla_cls.return_value = mock_vla_instance
        
        mock_model = MagicMock()
        mock_pg.from_pretrained.return_value = mock_model
        mock_prepare.return_value = mock_model
        mock_get_peft.return_value = mock_model
        
        mock_vla_instance.model = mock_model
        mock_vla_instance.processor = MagicMock()
        mock_vla_instance.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        
        # Mock forward pass output
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(1.0, requires_grad=True)
        mock_vla_instance.return_value = mock_output
        
        # Mock data
        data = [{
            'image': torch.randn(3, 224, 224),
            'prompt': 'test prompt',
            'text': 'test text',
            'state_vec': torch.randn(8),
            'actions': torch.randn(4)
        }] * 4
        
        # Run training
        train_vla(
            ckpt_path="dummy/ckpt",
            data=data,
            output_dir="dummy_output",
            epochs=1,
            batch_size=2
        )
        
        # Assertions
        mock_vla_cls.assert_called()
        mock_pg.from_pretrained.assert_called()
        mock_prepare.assert_called()
        mock_get_peft.assert_called()
        self.assertEqual(mock_vla_instance.call_count, 2)
        
    @patch('trainers.vla_trainer.VLA')
    @patch('trainers.vla_trainer.PeftModel')
    @patch('torch.load')
    def test_run_inference(self, mock_load, mock_peft, mock_vla_cls):
        # Setup mocks
        mock_vla_instance = MagicMock()
        mock_vla_cls.return_value = mock_vla_instance
        
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_peft.from_pretrained.return_value = mock_model
        mock_vla_instance.model = mock_model
        
        # Mock tensor returns
        mock_model.vision_tower.return_value.last_hidden_state = torch.randn(1, 10, 768)
        mock_model.multi_modal_projector.return_value = torch.randn(1, 10, 768)
        mock_vla_instance.state_projector.return_value = torch.randn(1, 1, 768)
        mock_model.get_input_embeddings.return_value.return_value = torch.randn(1, 5, 768)
        mock_model.config.hidden_size = 768
        
        # Mock processor return
        mock_inputs = MagicMock()
        mock_inputs.__getitem__.side_effect = lambda k: {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "input_ids": torch.randint(0, 100, (1, 5)),
            "attention_mask": torch.ones(1, 5)
        }[k]
        mock_inputs.to.return_value = mock_inputs
        mock_vla_instance.processor.return_value = mock_inputs
        
        # Mock generate output
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_vla_instance.processor.batch_decode.return_value = ["generated text"]
        
        # Run inference
        from trainers.vla_trainer import run_inference
        decoded_text, pred_actions = run_inference(
            ckpt_path="dummy/ckpt",
            adapter_path="dummy/adapter",
            image=torch.randn(3, 224, 224),
            prompt="test prompt",
            state_vec=torch.randn(8)
        )
        
        # Assertions
        # We expect generate to be called, not predict (which does forward pass)
        mock_model.generate.assert_called()

class TestRunSampleTraining(unittest.TestCase):
    @patch('trainers.run_sample_training.train_vla')
    def test_main(self, mock_train_vla):
        from trainers.run_sample_training import main
        
        # Mock sys.argv
        with patch('sys.argv', ['run_sample_training.py', '--epochs', '1', '--batch_size', '2']):
            main()
            
        # Verify train_vla was called
        mock_train_vla.assert_called_once()
        
        # Verify args passed to train_vla
        call_args = mock_train_vla.call_args
        self.assertEqual(call_args.kwargs['epochs'], 1)
        self.assertEqual(call_args.kwargs['batch_size'], 2)
        self.assertEqual(len(call_args.kwargs['data']), 4) # Default num_samples


        
    def test_collate_fn(self):
        batch = [{
            'image': torch.randn(3, 224, 224),
            'prompt': 'test prompt',
            'text': 'test text',
            'state_vec': torch.randn(8),
            'actions': torch.randn(4)
        }] * 2
        
        batch_out = collate_fn(batch)
        self.assertEqual(len(batch_out['image']), 2)
        self.assertEqual(batch_out['state_vec'].shape, (2, 8))
        self.assertEqual(batch_out['actions'].shape, (2, 4))

if __name__ == '__main__':
    unittest.main()
