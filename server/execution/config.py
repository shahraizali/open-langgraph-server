"""Configuration module for LangGraph execution."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExecutionConfigBuilder:
    """Builds execution configuration for LangGraph runs."""

    @staticmethod
    def build_execution_config(
        run_id: str,
        thread_id: str,
        run_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build execution configuration for LangGraph.
        
        Args:
            run_id: The run ID
            thread_id: The thread ID
            run_config: Optional run configuration from database
            
        Returns:
            Merged execution configuration
        """
        # Base configuration with required fields
        config = {
            "configurable": {
                "thread_id": thread_id,
                "run_id": run_id
            }
        }

        logger.info(f"[CONFIG] Base execution config: {config}")

        if run_config and isinstance(run_config, dict):
            logger.info(f"[CONFIG] Merging run config: {run_config}")
            logger.info(f"[CONFIG] Run config keys: {list(run_config.keys())}")

            # Merge configurable section
            if "configurable" in run_config and isinstance(run_config["configurable"], dict):
                logger.info(f"[CONFIG] Merging configurable: {run_config['configurable']}")
                config["configurable"].update(run_config["configurable"])
                logger.info(f"[CONFIG] Config after merging configurable: {config}")
            else:
                logger.warning(
                    f"[CONFIG] Skipping configurable merge. "
                    f"configurable exists: {'configurable' in run_config}, "
                    f"value: {run_config.get('configurable')}, "
                    f"type: {type(run_config.get('configurable'))}"
                )

            # Merge other top-level keys
            for key, value in run_config.items():
                if key != "configurable":
                    logger.info(f"[CONFIG] Merging top-level key '{key}': {value}")
                    config[key] = value
        else:
            logger.warning(
                f"[CONFIG] Skipping config merge. run_config: {run_config}, "
                f"type: {type(run_config)}"
            )

        logger.info(f"[CONFIG] Final execution config: {config}")
        ExecutionConfigBuilder._log_config_structure(config)

        return config

    @staticmethod
    def _log_config_structure(config: Dict[str, Any]) -> None:
        """Log detailed config structure for debugging."""
        logger.info("[CONFIG] Config structure details:")
        for key, value in config.items():
            logger.info(f"[CONFIG]   {key}: {value} (type: {type(value)})")
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    logger.info(f"[CONFIG]     {subkey}: {subvalue} (type: {type(subvalue)})")

    @staticmethod
    def prepare_stream_mode(stream_mode: Optional[list] = None) -> list:
        """Prepare and validate stream mode configuration.
        
        Args:
            stream_mode: Optional stream mode list
            
        Returns:
            Validated stream mode list
        """
        if not stream_mode:
            stream_mode = ["values"]

        # Patch: Always include 'messages' if 'messages-tuple' or 'custom' is present
        if ("messages-tuple" in stream_mode or "custom" in stream_mode) and "messages" not in stream_mode:
            stream_mode.append("messages")

        logger.info(f"[CONFIG] Stream mode: {stream_mode}")
        return stream_mode

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate execution configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(config, dict):
            logger.error(f"[CONFIG] Invalid config type: {type(config)}")
            return False

        if "configurable" not in config:
            logger.error("[CONFIG] Missing 'configurable' key in config")
            return False

        configurable = config["configurable"]
        if not isinstance(configurable, dict):
            logger.error(f"[CONFIG] Invalid configurable type: {type(configurable)}")
            return False

        required_keys = ["thread_id", "run_id"]
        for key in required_keys:
            if key not in configurable:
                logger.error(f"[CONFIG] Missing required key '{key}' in configurable")
                return False

        logger.info("[CONFIG] Configuration validation passed")
        return True