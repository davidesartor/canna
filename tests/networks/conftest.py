import pytest

from canna.networks.mmdit import MMDiTBlock

from ._helpers import mmditflow, perturbed, rngs

# module-scoped: construction and tracing dominate these tests, the forward passes do not.
# Every consumer is read-only -- anything that perturbs or mutates builds its own.


@pytest.fixture(scope="module")
def flow():
    return mmditflow()


@pytest.fixture(scope="module")
def perturbed_flow():
    return perturbed(mmditflow())


@pytest.fixture(scope="module")
def perturbed_block():
    return perturbed(MMDiTBlock(8, 2, 2, rngs=rngs()))
