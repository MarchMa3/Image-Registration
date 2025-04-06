import unittest
import torch
from brainmvp_adapter import BrainMVPAdapter
from unittest.mock import patch, MagicMock

class TestBrainMVPAdapter(unittest.TestCase):
    @patch('brainmvp_adapter.UniUnet')
    @patch('torch.load')
    def test_initialization(self, mock_torch_load, mock_uni_unet):
        # Mock the model and checkpoint
        mock_model = MagicMock()
        mock_uni_unet.return_value = mock_model
        mock_torch_load.return_value = {'state_dict': {}}
        
        # Initialize the adapter
        adapter = BrainMVPAdapter(
            brainmvp_checkpoint="fake_path.pt",
            in_channels=4,
            patch_shape=96,
            out_classes=3,
            batch_size=2
        )
        
        # Verify initialization parameters
        self.assertEqual(adapter.batch_size, 2)
        self.assertEqual(adapter.patch_shape, 96)
        
        # Verify model initialization
        mock_uni_unet.assert_called_once_with(
            input_shape=96, in_channels=4, out_channels=3
        )
        
        # Verify model loaded weights
        mock_torch_load.assert_called_once_with("fake_path.pt", map_location=torch.device('cpu'))
        mock_model.load_state_dict.assert_called_once()


    def test_get_transforms(self):
        adapter = BrainMVPAdapter(
            brainmvp_checkpoint="fake_path.pt",
            in_channels=4,
            patch_shape=96,
            out_classes=3
        )
        
        transforms = adapter._get_transforms()
        
        # Check that transforms is a MONAI Compose object
        from monai.transforms import Compose
        self.assertIsInstance(transforms, Compose)
        
        # Check that all expected transforms are included
        transform_names = [type(t).__name__ for t in transforms.transforms]
        expected_names = [
            "EnsureChannelFirstd", 
            "ScaleIntensityRangePercentilesd",
            "Orientationd", 
            "Spacingd", 
            "CropForegroundd"
        ]
        
        for name in expected_names:
            self.assertIn(name, transform_names)

    def test_extract_features(self):
        with patch('brainmvp_adapter.UniUnet') as mock_uni_unet:
            # Create mock model with mock encoder
            mock_model = MagicMock()
            mock_encoder = MagicMock()
            mock_encoder.return_value = torch.ones(2, 512, 3, 3, 3)  # Fake feature tensor
            mock_model.encoder = mock_encoder
            mock_uni_unet.return_value = mock_model
            
            # Create adapter
            adapter = BrainMVPAdapter(
                brainmvp_checkpoint="fake_path.pt",
                in_channels=4,
                patch_shape=96,
                out_classes=3
            )
            
            # Create fake 3D image
            fake_image = torch.ones(2, 4, 96, 96, 96)  # [B, C, H, W, D]
            
            # Extract features
            with torch.no_grad():
                features = adapter.extract_features(fake_image)
                
            # Verify encoder was called with the image
            mock_encoder.assert_called_once()
            self.assertTrue(torch.all(features == torch.ones(2, 512, 3, 3, 3)))

    def test_resize_to_patch_shape(self):
        with patch('brainmvp_adapter.UniUnet'):
            adapter = BrainMVPAdapter(
                brainmvp_checkpoint="fake_path.pt",
                in_channels=4,
                patch_shape=96,
                out_classes=3
            )
            
            # Test case 1: Image already has correct shape
            image = torch.ones(2, 4, 96, 96, 96)
            resized = adapter._resize_to_patch_shape(image)
            self.assertEqual(resized.shape, (2, 4, 96, 96, 96))
            
            # Test case 2: Image needs resizing
            image = torch.ones(2, 4, 64, 64, 64)
            with patch('scipy.ndimage.zoom', return_value=np.ones((96, 96, 96))):
                resized = adapter._resize_to_patch_shape(image)
                self.assertEqual(resized.shape, (2, 4, 96, 96, 96))
    
    def test_process_for_atom(self):
        with patch('brainmvp_adapter.UniUnet'):
            adapter = BrainMVPAdapter(
                brainmvp_checkpoint="fake_path.pt",
                in_channels=4,
                patch_shape=96,
                out_classes=3
            )
            
            # Create fake 3D image input
            fake_image = torch.ones(2, 4, 20, 20, 20)
            
            # Mock the extraction methods
            adapter._resize_to_uniform_size = MagicMock(return_value=fake_image)
            
            # Create expected outputs
            with patch('random.randint', side_effect=[10, 10, 10, 10, 10, 10]):
                with patch('brainmvp_adapter.extract_slice', return_value=(
                    torch.zeros(4, 4), [0, 0, 0], [[0, 0], [0, 0], [0, 0]]
                )):
                    with patch('brainmvp_adapter.getposition_1', return_value=[0, 1]):
                        with patch('brainmvp_adapter.getposition_2', return_value=[0, 1, 2]):
                            # Call the method
                            resized_imgs, img_2d, label1, label1_loss1, label2, label2_loss = adapter.process_for_atom(fake_image)
                            
                            # Verify outputs
                            self.assertEqual(resized_imgs.shape, fake_image.shape)
                            self.assertEqual(img_2d.shape, (2, 4, 4))
                            self.assertEqual(len(label1), 2)
                            self.assertEqual(label1_loss1.shape, (2, 27))
                            self.assertEqual(len(label2), 2)
                            self.assertEqual(len(label2_loss), 2)
    
    def test_resize_to_uniform_size(self):
        with patch('brainmvp_adapter.UniUnet'):
            adapter = BrainMVPAdapter(
                brainmvp_checkpoint="fake_path.pt",
                in_channels=4,
                patch_shape=96,
                out_classes=3
            )
            
            # Create fake input
            image = torch.ones(2, 4, 96, 96, 96)
            
            # Test with scipy.ndimage.zoom mocked
            with patch('scipy.ndimage.zoom', return_value=np.ones((20, 20, 20))):
                resized = adapter._resize_to_uniform_size(image, target_size=(20, 20, 20))
                self.assertEqual(resized.shape, (2, 4, 20, 20, 20))

    def test_process_batch(self):
        with patch('brainmvp_adapter.UniUnet'):
            adapter = BrainMVPAdapter(
                brainmvp_checkpoint="fake_path.pt",
                in_channels=4,
                patch_shape=96,
                out_classes=3
            )
            
            # Mock transform and other methods
            adapter.transforms = MagicMock(return_value={"image": torch.ones(2, 4, 96, 96, 96)})
            adapter.extract_features = MagicMock(return_value=torch.ones(2, 512, 3, 3, 3))
            adapter.process_for_atom = MagicMock(return_value=(
                torch.ones(2, 4, 20, 20, 20),  # resized_imgs
                torch.ones(2, 4, 4),          # img_2d
                [[0, 1], [0, 1]],             # label1
                torch.ones(2, 27),            # label1_loss1
                [[[0, 1, 2]], [[0, 1, 2]]],   # label2
                [[[0, 1, 0]], [[0, 1, 0]]]    # label2_loss
            ))
            
            # Create fake batch data
            batch_data = {"image": torch.ones(2, 4, 96, 96, 96)}
            
            # Call process_batch
            result = adapter.process_batch(batch_data)
            
            # Check that methods were called
            adapter.transforms.assert_called_once_with(batch_data)
            adapter.extract_features.assert_called_once()
            adapter.process_for_atom.assert_called_once()
            
            # Verify output structure
            self.assertEqual(len(result), 6)
    
    def test_process_batch(self):
        with patch('brainmvp_adapter.UniUnet'):
            adapter = BrainMVPAdapter(
                brainmvp_checkpoint="fake_path.pt",
                in_channels=4,
                patch_shape=96,
                out_classes=3
            )
            
            # Mock transform and other methods
            adapter.transforms = MagicMock(return_value={"image": torch.ones(2, 4, 96, 96, 96)})
            adapter.extract_features = MagicMock(return_value=torch.ones(2, 512, 3, 3, 3))
            adapter.process_for_atom = MagicMock(return_value=(
                torch.ones(2, 4, 20, 20, 20),  # resized_imgs
                torch.ones(2, 4, 4),          # img_2d
                [[0, 1], [0, 1]],             # label1
                torch.ones(2, 27),            # label1_loss1
                [[[0, 1, 2]], [[0, 1, 2]]],   # label2
                [[[0, 1, 0]], [[0, 1, 0]]]    # label2_loss
            ))
            
            # Create fake batch data
            batch_data = {"image": torch.ones(2, 4, 96, 96, 96)}
            
            # Call process_batch
            result = adapter.process_batch(batch_data)
            
            # Check that methods were called
            adapter.transforms.assert_called_once_with(batch_data)
            adapter.extract_features.assert_called_once()
            adapter.process_for_atom.assert_called_once()
            
            # Verify output structure
            self.assertEqual(len(result), 6)
    
    @patch('brainmvp_adapter.BrainMVPAdapter')
    def test_data_generator_init(self, mock_adapter_class):
        # Mock adapter
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter
        
        # Create data generator
        generator = BrainMVPDataGenerator(
            img_dir="/fake/path",
            split="train",
            brainmvp_checkpoint="fake_checkpoint.pt",
            batch_size=4,
            in_channels=4,
            out_classes=3
        )
        
        # Verify BrainMVPAdapter was initialized correctly
        mock_adapter_class.assert_called_once_with(
            brainmvp_checkpoint="fake_checkpoint.pt",
            in_channels=4,
            out_classes=3,
            batch_size=4
        )
        
        # Check initialization values
        self.assertEqual(generator.img_dir, "/fake/path")
        self.assertEqual(generator.split, "train")
        self.assertEqual(generator.batch_size, 4)
        self.assertEqual(generator.dim, (20, 20, 20))
        self.assertEqual(generator.dim2d, (4, 4))
        self.assertEqual(generator.dimlabel1, (27,))

    @patch('brainmvp_adapter.MRIDataGenerator')
    def test_create_mri(self, mock_mri_generator):
        # Mock MRIDataGenerator
        mock_generator = MagicMock()
        mock_generator.imaged = np.zeros((1, 200, 200, 200))
        mock_mri_generator.return_value = mock_generator
        
        # Create data generator
        generator = BrainMVPDataGenerator(
            img_dir="/fake/path",
            split="train",
            brainmvp_checkpoint="fake_checkpoint.pt"
        )
        
        # Call create_MRI
        generator.create_MRI()
        
        # Verify MRIDataGenerator was created correctly
        mock_mri_generator.assert_called_once_with("/fake/path", "train")
        
        # Check that imaged was set correctly
        self.assertTrue(hasattr(generator, 'imaged'))

    def test_integration(self):
        # This tests the full flow from initialization to processing a batch
        
        # Mock all external dependencies
        with patch('brainmvp_adapter.UniUnet') as mock_uni_unet, \
            patch('torch.load') as mock_torch_load, \
            patch('brainmvp_adapter.extract_slice') as mock_extract_slice, \
            patch('brainmvp_adapter.getposition_1') as mock_getposition_1, \
            patch('brainmvp_adapter.getposition_2') as mock_getposition_2, \
            patch('scipy.ndimage.zoom', return_value=np.ones((20, 20, 20))):
            
            # Setup mocks
            mock_model = MagicMock()
            mock_encoder = MagicMock()
            mock_encoder.return_value = torch.ones(2, 512, 3, 3, 3)
            mock_model.encoder = mock_encoder
            mock_uni_unet.return_value = mock_model
            mock_torch_load.return_value = {'state_dict': {}}
            
            mock_extract_slice.return_value = (torch.zeros(4, 4), [0, 0, 0], [[0, 0], [0, 0], [0, 0]])
            mock_getposition_1.return_value = [0, 1]
            mock_getposition_2.return_value = [0, 1, 2]
            
            # Create adapter
            adapter = BrainMVPAdapter(
                brainmvp_checkpoint="fake_path.pt",
                in_channels=4,
                patch_shape=96,
                out_classes=3,
                batch_size=2
            )
            
            # Create fake batch data
            batch_data = {"image": torch.ones(2, 4, 96, 96, 96)}
            
            # Process batch
            with patch('random.randint', return_value=10):
                results = adapter.process_batch(batch_data)
                
                # Verify results structure
                self.assertEqual(len(results), 6)
                resized_imgs, img_2d, label1, label1_loss1, label2, label2_loss = results
                
                # Basic shape checks
                self.assertEqual(resized_imgs.shape[:2], (2, 4))  # Batch size and channels