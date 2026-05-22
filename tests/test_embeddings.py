from __future__ import annotations

import pytest
import torch

from visu_predict.models.embeddings import (
    AdaptiveEmbedding,
    DayOfWeekEmbedding,
    NodeEmbedding,
    STAEInputComposer,
    TimeOfDayEmbedding,
)


def test_adaptive_embedding_shape_and_lookup():
    emb = AdaptiveEmbedding(steps_per_day=288, num_sensors=10, d_adp=16)
    assert emb.weight.shape == (288, 10, 16)
    tod = torch.randint(0, 288, (2, 5))  # [B=2, T=5]
    out = emb(tod)
    assert out.shape == (2, 5, 10, 16)


def test_adaptive_embedding_gradient_flows():
    emb = AdaptiveEmbedding(steps_per_day=12, num_sensors=4, d_adp=8)
    tod = torch.tensor([[0, 1, 2]])
    out = emb(tod)
    out.sum().backward()
    assert emb.weight.grad is not None
    assert (emb.weight.grad != 0).any()


def test_time_of_day_embedding_table_size():
    emb = TimeOfDayEmbedding(steps_per_day=96, d_tod=12)
    assert emb.num_embeddings == 96
    assert emb.embedding_dim == 12
    out = emb(torch.tensor([0, 95]))
    assert out.shape == (2, 12)


def test_day_of_week_embedding_table_size():
    emb = DayOfWeekEmbedding(d_dow=8)
    assert emb.num_embeddings == 7
    out = emb(torch.tensor([0, 6]))
    assert out.shape == (2, 8)


def test_node_embedding_broadcasts():
    emb = NodeEmbedding(num_sensors=5, d_node=4)
    out = emb(batch_size=3, seq_length=7)
    assert out.shape == (3, 7, 5, 4)


def test_stae_input_composer_full_shape():
    composer = STAEInputComposer(
        steps_per_day=288, num_sensors=5,
        d_input=4, d_tod=6, d_dow=8, d_adp=10, d_node=4,
    )
    assert composer.d_model == 32
    traffic = torch.randn(2, 6, 5)
    tod = torch.randint(0, 288, (2, 6))
    dow = torch.randint(0, 7, (2, 6))
    out = composer(traffic, tod, dow)
    assert out.shape == (2, 6, 5, 32)


def test_stae_input_composer_d_input_zero_raises():
    with pytest.raises(ValueError, match="d_input must be positive"):
        STAEInputComposer(
            steps_per_day=288, num_sensors=5,
            d_input=0, d_tod=8, d_dow=8, d_adp=8, d_node=8,
        )


def test_stae_input_composer_optional_components_zero_dim():
    composer = STAEInputComposer(
        steps_per_day=288, num_sensors=3,
        d_input=8, d_tod=0, d_dow=0, d_adp=8, d_node=0,
    )
    traffic = torch.randn(1, 4, 3)
    tod = torch.zeros(1, 4, dtype=torch.long)
    dow = torch.zeros(1, 4, dtype=torch.long)
    out = composer(traffic, tod, dow)
    assert out.shape == (1, 4, 3, 16)
