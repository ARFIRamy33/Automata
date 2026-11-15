
from pathlib import Path

from automata_extensions.fa.dfa import ExtendedDFA


def test_minimisation_avec_graphviz():
    automate = ExtendedDFA(
        states={"A", "B", "C", "D", "E"},
        input_symbols={"a", "b"},
        transitions={
            "A": {"a": "B", "b": "D"},
            "B": {"a": "C", "b": "E"},
            "C": {"a": "C", "b": "E"},
            "D": {"a": "E", "b": "C"},
            "E": {"a": "E", "b": "C"},
        },
        initial_state="A",
        final_states={"C", "E"},
        allow_partial=False,
    )

    # Dossier de sortie
    dossier = Path("resultats_graphviz")
    dossier.mkdir(exist_ok=True)

   
    automate.svg(dossier / "automate_initial.svg")

    
    automate_minimal = automate.minimize()

    
    automate_minimal.svg(dossier / "automate_minimise.svg")

   
    assert len(automate_minimal.states) == 3
    assert automate_minimal.is_minimal()

    print("\nAutomate initial :", dossier / "automate_initial.svg")
    print("Automate minimisé :", dossier / "automate_minimise.svg")
    print("Test réussi !")
