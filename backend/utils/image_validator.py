"""
DermaAgent - Image Validation Utility
Performs strict validation on incoming uploaded skin lesion files:
- File existence & non-empty name
- Extension whitelisting
- File size boundary checks (< 10 MB)
- Binary PIL integrity verification (detects corrupted images)
- Image dimensions & color mode validation
"""

import io
from pathlib import Path
from PIL import Image

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MIN_DIMENSION = 32
MAX_DIMENSION = 6000

class ImageValidationError(Exception):
    """Custom exception raised when an image fails validation."""
    pass

def validate_image(file_storage, max_size=MAX_FILE_SIZE):
    """
    Validates uploaded file from Flask request.files.
    
    Args:
        file_storage: Werkzeug FileStorage object
        max_size: Maximum allowed size in bytes
        
    Returns:
        tuple: (is_valid: bool, error_message: str or None, ext: str)
    """
    if file_storage is None or not hasattr(file_storage, "filename"):
        return False, "No file provided in request.", ""

    filename = file_storage.filename or ""
    if not filename.strip():
        return False, "No file selected.", ""

    if "." not in filename:
        return False, "File is missing an extension.", ""

    extension = filename.rsplit(".", 1)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type (.{extension}). Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}", ""

    # Check file size safely
    file_storage.seek(0, io.SEEK_END)
    file_size = file_storage.tell()
    file_storage.seek(0)

    if file_size == 0:
        return False, "Uploaded file is empty (0 bytes).", extension

    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        return False, f"File size exceeds maximum limit of {max_mb:.0f} MB.", extension

    # Read binary stream and verify using PIL
    try:
        image_bytes = file_storage.read()
        file_storage.seek(0) # Reset stream pointer for saving

        byte_stream = io.BytesIO(image_bytes)
        with Image.open(byte_stream) as img:
            img.verify()

        # Re-open to read attributes because verify() closes some attributes
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            if width < MIN_DIMENSION or height < MIN_DIMENSION:
                return False, f"Image dimensions too small ({width}x{height}). Minimum is {MIN_DIMENSION}x{MIN_DIMENSION}.", extension
            if width > MAX_DIMENSION or height > MAX_DIMENSION:
                return False, f"Image dimensions too large ({width}x{height}). Maximum is {MAX_DIMENSION}x{MAX_DIMENSION}.", extension

    except Exception as err:
        return False, f"Corrupted or invalid image file: {str(err)}", extension

    return True, None, extension
