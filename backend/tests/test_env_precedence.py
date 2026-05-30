import os
import unittest
from pathlib import Path
import importlib
import sys
from unittest.mock import MagicMock, patch


sys.modules["dotenv"] = MagicMock()

class TestEnvPrecedence(unittest.TestCase):
    @patch("pathlib.Path.exists")
    def test_env_loading_order(self, mock_exists):
        """
        Verify that dotenv files are discovered in the expected precedence: root .env, then backend/.env, then services/.env.
        
        Reloads app.services.config with pathlib.Path.exists forced to True, reads config._dotenv_paths, and asserts that a root `.env`, a `backend/.env` (excluding paths containing "app"), and a `services/.env` are present and ordered such that backend/.env appears after the root .env and services/.env appears after backend/.env.
        
        Parameters:
            mock_exists (unittest.mock.Mock): Patched replacement for pathlib.Path.exists; this test sets its return_value to True to simulate that all candidate .env files exist.
        """
        # Force exists() to return True for all .env files
        mock_exists.return_value = True
        
        import app.services.config as config
        importlib.reload(config)
        
        paths = config._dotenv_paths
        
        # Check that we have at least root and backend paths in the correct order
        root_idx = -1
        backend_idx = -1
        services_idx = -1
        
        for i, p in enumerate(paths):
            p_str = str(p).lower()
            if "backend" in p_str and ".env" in p_str and "app" not in p_str:
                backend_idx = i
            elif "services" in p_str and ".env" in p_str:
                services_idx = i
            elif ".env" in p_str:
                root_idx = i
        
        self.assertNotEqual(root_idx, -1, "Root .env path not found in loading list")
        self.assertNotEqual(backend_idx, -1, "Backend .env path not found in loading list")
        self.assertNotEqual(services_idx, -1, "Services .env path not found in loading list")
        
        self.assertGreater(backend_idx, root_idx, "backend/.env must be loaded after root .env to override it")
        self.assertGreater(services_idx, backend_idx, "services/.env should have highest priority")

if __name__ == "__main__":
    unittest.main()
