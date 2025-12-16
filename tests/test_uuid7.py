"""
Tests for UUIDv7 generator.

Validates RFC 9562 compliance:
- Returns valid UUID objects
- Version is 7
- UUIDs are unique
- UUIDs are roughly time-ordered
"""

import time
from uuid import UUID

import pytest

from app.domain.utils.uuid7 import uuid7


class TestUuid7Basic:
    """Basic tests for uuid7() function."""

    def test_returns_uuid_object(self):
        """uuid7() should return a UUID instance."""
        result = uuid7()
        assert isinstance(result, UUID)

    def test_uuid_version_is_7(self):
        """Generated UUID should have version 7."""
        result = uuid7()
        assert result.version == 7

    def test_uuid_variant_is_rfc4122(self):
        """Generated UUID should have RFC 4122 variant."""
        result = uuid7()
        # Variant bits should be 10xx (RFC 4122)
        # UUID.variant returns the variant as a string
        assert result.variant == "specified in RFC 4122" or str(result.variant) == "RFC_4122"

    def test_multiple_calls_produce_unique_uuids(self):
        """Each call should produce a unique UUID."""
        uuids = [uuid7() for _ in range(100)]
        unique_uuids = set(uuids)
        assert len(unique_uuids) == 100

    def test_uuid_is_valid_string_format(self):
        """UUID should have valid string representation."""
        result = uuid7()
        uuid_str = str(result)
        
        # Should be 36 characters (8-4-4-4-12 with hyphens)
        assert len(uuid_str) == 36
        
        # Should be parseable back to UUID
        parsed = UUID(uuid_str)
        assert parsed == result


class TestUuid7TimeOrdering:
    """Tests for time-ordering property of UUIDv7."""

    def test_uuids_are_time_ordered(self):
        """UUIDs generated later should sort after earlier ones."""
        uuids = []
        
        # Generate 5 UUIDs with small delays to ensure different timestamps
        for _ in range(5):
            uuids.append(uuid7())
            time.sleep(0.002)  # 2ms delay to ensure different millisecond timestamps
        
        # Convert to strings for comparison (UUIDv7 sorts correctly as strings)
        uuid_strings = [str(u) for u in uuids]
        
        # Should already be sorted (time-ordered)
        assert uuid_strings == sorted(uuid_strings)

    def test_uuids_generated_same_millisecond_are_unique(self):
        """UUIDs generated in same millisecond should still be unique."""
        # Generate many UUIDs as fast as possible
        uuids = [uuid7() for _ in range(1000)]
        unique_uuids = set(uuids)
        
        # All should be unique despite potential same-millisecond generation
        assert len(unique_uuids) == 1000

    def test_uuid_bytes_contain_timestamp(self):
        """First 48 bits should contain Unix timestamp in milliseconds."""
        before_ms = int(time.time() * 1000)
        result = uuid7()
        after_ms = int(time.time() * 1000)
        
        # Extract timestamp from first 6 bytes
        uuid_bytes = result.bytes
        timestamp_bytes = uuid_bytes[:6]
        extracted_timestamp = int.from_bytes(timestamp_bytes, byteorder="big")
        
        # Timestamp should be within the time window
        assert before_ms <= extracted_timestamp <= after_ms + 1


class TestUuid7Consistency:
    """Tests for consistency and reproducibility."""

    def test_uuid_bytes_length(self):
        """UUID should be exactly 16 bytes."""
        result = uuid7()
        assert len(result.bytes) == 16

    def test_version_bits_are_correct(self):
        """Bits 48-51 should be 0111 (version 7)."""
        result = uuid7()
        uuid_bytes = result.bytes
        
        # Byte 6 high nibble should be 0111 (0x7)
        version_nibble = (uuid_bytes[6] >> 4) & 0x0F
        assert version_nibble == 7

    def test_variant_bits_are_correct(self):
        """Bits 64-65 should be 10 (RFC 4122 variant)."""
        result = uuid7()
        uuid_bytes = result.bytes
        
        # Byte 8 high 2 bits should be 10
        variant_bits = (uuid_bytes[8] >> 6) & 0x03
        assert variant_bits == 2  # Binary 10
