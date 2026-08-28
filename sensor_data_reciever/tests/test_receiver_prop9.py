"""
Property-based test for generate_record_id — Property 9.

**Validates: Requirements 4.2**

Property 9: generate_record_id produces sequential R-prefixed IDs.

For any positive integer N, calling generate_record_id exactly N times from
record_counter=0 produces "R000001", "R000002", ..., "R{N:06d}" in order with
no gaps or repetitions.
"""

import receiver
from hypothesis import given, settings
import hypothesis.strategies as st


@given(st.integers(min_value=1, max_value=200))
@settings(max_examples=200)
def test_generate_record_id_produces_sequential_r_prefixed_ids(n):
    """
    Property 9: generate_record_id produces sequential R-prefixed IDs.

    For any positive integer N, calling generate_record_id N times from a
    fresh (counter=0) state returns exactly ["R000001", "R000002", ...,
    f"R{N:06d}"] with no gaps or repetitions.
    """
    # Reset module-level counter to ensure isolation between hypothesis examples
    receiver.record_counter = 0

    expected = [f"R{i:06d}" for i in range(1, n + 1)]
    actual = [receiver.generate_record_id() for _ in range(n)]

    assert actual == expected, (
        f"For N={n}: expected {expected[:5]}{'...' if n > 5 else ''}, "
        f"got {actual[:5]}{'...' if n > 5 else ''}"
    )
