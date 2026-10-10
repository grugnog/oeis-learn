from oeis_learn.decoder.recurrence import order2_template, recognize, recognize_grounded
from oeis_learn.decoder.grounding import ground, splice


def terms():
    out = [0, 1]
    for _ in range(18):
        out.append(sum(out[-2:]))
    return out


def test_alpha_renamed_and_interleaved_state_transfers():
    source = order2_template()
    for src, dst in (("b", "a"), ("t", "b")):
        grouped = (
            " ".join(f"local.get ${src}{j}" for j in range(4))
            + " "
            + " ".join(f"local.set ${dst}{j}" for j in reversed(range(4)))
        )
        paired = " ".join(f"local.get ${src}{j} local.set ${dst}{j}" for j in range(4))
        source = source.replace(grouped, paired)
    source = (
        source.replace("$a", "$previous")
        .replace("$b", "$current")
        .replace("$t", "$next")
        .replace("$i", "$counter")
        .replace("$n", "$index")
        .replace("$exit", "$done")
        .replace("$loop", "$again")
    )
    assert recognize(source) == (0, 1)
    result = ground(source, terms())
    assert result.outcome == "verified_solution" and result.constants == (1, 1), result
    assert recognize_grounded(result.source) == ((0, 1), (1, 1))
    assert recognize(source.replace("i32.ge_s", "i32.gt_s")) is None


def test_wrong_state_transfer_and_initial_condition():
    source = order2_template()
    assert recognize(source.replace("local.get $t0", "local.get $a0")) is None
    assert ground(order2_template(1, 1), terms()).outcome == "unsupported"
