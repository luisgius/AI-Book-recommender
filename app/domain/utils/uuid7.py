"""
UUIDv7 generator following RFC 9562.

=============================================================================
TEACHING NOTES: Why UUIDv7 over UUIDv4?
=============================================================================

UUIDv4 (random):
- Completely random 122 bits
- No inherent ordering
- Poor database index locality (random inserts scatter across B-tree)

UUIDv7 (time-ordered):
- First 48 bits: Unix timestamp in milliseconds
- Next 4 bits: version (0111 = 7)
- Next 12 bits: random (sub-millisecond ordering)
- Next 2 bits: variant (10)
- Last 62 bits: random

Benefits of UUIDv7:
1. Time-ordered: UUIDs generated later sort after earlier ones
2. Better DB locality: sequential inserts cluster in B-tree pages
3. Implicit creation timestamp: first 48 bits encode when it was created
4. Still globally unique: random bits prevent collisions
5. Compatible: same UUID type, same storage, same parsing

For our book catalog:
- Books ingested together get sequential IDs
- SQLite index performance improves
- Debugging is easier (IDs reveal creation order)

RFC 9562: https://www.rfc-editor.org/rfc/rfc9562.html
=============================================================================
"""

import os
import time
from uuid import UUID


def uuid7() -> UUID:
    """
    Generate a UUIDv7 (time-ordered) following RFC 9562.

    Structure (128 bits total):
    - Bits 0-47 (48 bits): Unix timestamp in milliseconds
    - Bits 48-51 (4 bits): Version = 0111 (7)
    - Bits 52-63 (12 bits): Random (rand_a)
    - Bits 64-65 (2 bits): Variant = 10
    - Bits 66-127 (62 bits): Random (rand_b)

    Returns:
        A uuid.UUID instance with version 7.

    Example:
        >>> from app.domain.utils.uuid7 import uuid7
        >>> book_id = uuid7()
        >>> isinstance(book_id, UUID)
        True
        >>> book_id.version
        7
    """
    # Get current Unix timestamp in milliseconds (48 bits)
    timestamp_ms = int(time.time() * 1000)

    # Generate 10 random bytes (80 bits) for rand_a (12 bits) + rand_b (62 bits)
    # We need 74 random bits total, but we'll generate 80 and mask appropriately
    random_bytes = os.urandom(10)

    # Build the 128-bit UUID:
    # Bytes 0-5: timestamp_ms (48 bits, big-endian)
    # Byte 6: version (4 bits) + rand_a high (4 bits)
    # Byte 7: rand_a low (8 bits) - but we only use 12 bits total for rand_a
    # Bytes 8-15: variant (2 bits) + rand_b (62 bits)

    # Extract timestamp bytes (big-endian, 6 bytes)
    ts_bytes = timestamp_ms.to_bytes(6, byteorder="big")

    # Byte 6: version (0111) in high nibble + 4 random bits in low nibble
    byte_6 = 0x70 | (random_bytes[0] & 0x0F)

    # Byte 7: 8 random bits (completes the 12-bit rand_a)
    byte_7 = random_bytes[1]

    # Byte 8: variant (10) in high 2 bits + 6 random bits
    byte_8 = 0x80 | (random_bytes[2] & 0x3F)

    # Bytes 9-15: 56 random bits (7 bytes)
    rand_b_bytes = random_bytes[3:10]

    # Assemble the 16-byte UUID
    uuid_bytes = ts_bytes + bytes([byte_6, byte_7, byte_8]) + rand_b_bytes

    return UUID(bytes=uuid_bytes)
