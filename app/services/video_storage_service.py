import io
import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Any, Optional
import cloudinary
import cloudinary.uploader
import cloudinary.api
from app.core.config import settings

logger = logging.getLogger("recruitai-video-storage")


class BaseVideoStorageProvider(ABC):
    """
    Abstract interface for video storage providers (Cloudinary, Azure Blob Storage, S3, etc.)
    """

    @abstractmethod
    async def upload_video(
        self,
        file_content: bytes | BinaryIO,
        filename: str,
        content_type: str = "video/webm",
        folder: str = "recruitai_proctoring_recordings",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Uploads video data and returns storage metadata (public_id, url, secure_url, bytes, etc.)
        """
        pass

    @abstractmethod
    async def get_video_url(self, public_id: str) -> Optional[str]:
        """
        Retrieves a streamable or playable URL for the given video resource.
        """
        pass

    @abstractmethod
    async def delete_video(self, public_id: str) -> bool:
        """
        Deletes a video from the storage provider.
        """
        pass


class CloudinaryStorageProvider(BaseVideoStorageProvider):
    """
    Cloudinary implementation of BaseVideoStorageProvider.
    Uses environment variables for credentials and uploads video streams safely.
    """

    def __init__(
        self,
        cloud_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        self.cloud_name = cloud_name or settings.CLOUDINARY_CLOUD_NAME or os.getenv("CLOUDINARY_CLOUD_NAME", "")
        self.api_key = api_key or settings.CLOUDINARY_API_KEY or os.getenv("CLOUDINARY_API_KEY", "")
        self.api_secret = api_secret or settings.CLOUDINARY_API_SECRET or os.getenv("CLOUDINARY_API_SECRET", "")

        if not (self.cloud_name and self.api_key and self.api_secret):
            logger.warning(
                "Cloudinary credentials incomplete. Video uploads will fail unless configured in environment."
            )

        cloudinary.config(
            cloud_name=self.cloud_name,
            api_key=self.api_key,
            api_secret=self.api_secret,
            secure=True
        )

    def _sync_upload(
        self,
        file_content: bytes | BinaryIO,
        public_id: str,
        folder: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous upload execution executed in a worker thread.
        """
        context = {}
        if metadata:
            context = {k: str(v) for k, v in metadata.items() if v is not None}

        upload_params = {
            "resource_type": "auto",
            "folder": folder,
            "public_id": public_id,
            "overwrite": True,
            "chunk_size": 6000000, # 6MB chunks for large video files
            "context": context
        }

        # upload_large expects an open file-like object with .read() or a filename path
        file_to_upload = io.BytesIO(file_content) if isinstance(file_content, (bytes, bytearray)) else file_content

        response = cloudinary.uploader.upload_large(
            file_to_upload,
            **upload_params
        )
        return response

    async def upload_video(
        self,
        file_content: bytes | BinaryIO,
        filename: str,
        content_type: str = "video/webm",
        folder: str = "recruitai_proctoring_recordings",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            # Extract stem without extension for Cloudinary public_id
            public_id = filename.rsplit(".", 1)[0] if "." in filename else filename
            
            logger.info(f"Initiating Cloudinary upload for video '{public_id}' in folder '{folder}'...")
            
            response = await asyncio.to_thread(
                self._sync_upload,
                file_content,
                public_id,
                folder,
                metadata
            )
            
            secure_url = response.get("secure_url") or response.get("url")
            storage_public_id = response.get("public_id")
            bytes_size = response.get("bytes")
            duration = response.get("duration")
            format_name = response.get("format")

            logger.info(
                f"Cloudinary upload successful for {public_id}: "
                f"URL={secure_url}, Size={bytes_size} bytes, Duration={duration}s"
            )

            return {
                "public_id": storage_public_id,
                "url": secure_url,
                "secure_url": secure_url,
                "bytes": bytes_size,
                "duration": duration,
                "format": format_name,
                "provider": "cloudinary",
                "raw_response": response
            }
        except Exception as e:
            logger.error(f"Cloudinary video upload failed for {filename}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Cloudinary upload error: {str(e)}")

    async def get_video_url(self, public_id: str) -> Optional[str]:
        try:
            res = await asyncio.to_thread(
                cloudinary.api.resource,
                public_id,
                resource_type="video"
            )
            return res.get("secure_url") or res.get("url")
        except Exception as e:
            logger.warning(f"Failed to fetch Cloudinary video resource {public_id}: {e}")
            return None

    async def delete_video(self, public_id: str) -> bool:
        try:
            res = await asyncio.to_thread(
                cloudinary.uploader.destroy,
                public_id,
                resource_type="video"
            )
            return res.get("result") == "ok"
        except Exception as e:
            logger.error(f"Failed to delete Cloudinary video {public_id}: {e}")
            return False


class VideoStorageService:
    """
    Facade service orchestrating video uploads and storage operations.
    Can dynamically instantiate CloudinaryStorageProvider or future AzureBlobStorageProvider.
    """

    def __init__(self, provider_type: Optional[str] = None):
        selected_provider = (provider_type or settings.VIDEO_STORAGE_PROVIDER or "cloudinary").lower()
        if selected_provider == "cloudinary":
            self.provider: BaseVideoStorageProvider = CloudinaryStorageProvider()
        else:
            logger.warning(f"Unknown storage provider '{selected_provider}', defaulting to Cloudinary.")
            self.provider = CloudinaryStorageProvider()

    async def upload_recording(
        self,
        file_content: bytes | BinaryIO,
        filename: str,
        content_type: str = "video/webm",
        folder: str = "recruitai_proctoring_recordings",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return await self.provider.upload_video(
            file_content=file_content,
            filename=filename,
            content_type=content_type,
            folder=folder,
            metadata=metadata
        )

    async def get_recording_url(self, public_id: str) -> Optional[str]:
        return await self.provider.get_video_url(public_id)

    async def delete_recording(self, public_id: str) -> bool:
        return await self.provider.delete_video(public_id)


# Singleton instance
video_storage_service = VideoStorageService()
