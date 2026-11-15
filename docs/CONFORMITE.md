# Conformité des 62 fonctionnalités (audit du prompt 22)

Légende : **Conforme** = implémentée, algorithme vérifié contre un oracle indépendant, tests présents.
**Partielle** = une partie de l'énoncé n'est pas couverte ou pas vérifiée. Aucune fonction « Non implémentée ».

**Réserve générale** : toute la vérification a été faite avec une bibliothèque `automata` de substitution
(stub), pas avec `automata-lib` 9.2.0 réelle (non téléchargeable dans l'environnement d'audit). Les 63 tests
« native » sont ignorés et doivent être exécutés sur votre machine.

| N° | Fonctionnalité | Méthode | Implémentée | Testée | Statut |
|---|---|---|---|---|---|
| 1 | DFS itératif / récursif | `dfs_iterative`, `dfs_recursive` | oui | oui | Conforme |
| 2 | BFS | `bfs` | oui | oui | Conforme |
| 3 | États accessibles | `accessible_states` | oui | oui | Conforme |
| 4 | État accessible ? | `is_accessible` | oui | oui | Conforme |
| 5 | États coaccessibles | `coaccessible_states` | oui | oui | Conforme |
| 6 | État coaccessible ? | `is_coaccessible` | oui | oui | Conforme |
| 7 | États utiles | `useful_states` | oui | oui | Conforme |
| 8 | Trim | `trim` | oui | oui | Conforme (langage vide : 1 état, convention documentée) |
| 9 | Est émondé | `is_trim` | oui | oui | Conforme (même convention) |
| 10 | Composantes fortement connexes | `strongly_connected_components` | oui | oui | Conforme (Kosaraju) |
| 11 | Cycle | `has_cycle` | oui | oui | Conforme |
| 12 | Cycle depuis un état | `has_cycle_from` | oui | oui | Conforme |
| 13 | Langage vide | `is_empty` | oui | oui | Conforme |
| 14 | Langage infini | `is_infinite` | oui | oui | Conforme |
| 15 | Reconnaissance d'un mot | `accepts` | oui | oui | Partielle : DFA vérifié ; pour un NFA l'appel natif `accepts_input` n'a pas été exécuté |
| 16 | Trace d'exécution | `execution_trace` | **corrigée** | oui | Conforme (DFA : états ; NFA : ensembles) après correction |
| 17 | Atteignables depuis un état | `reachable_from` | oui | oui | Conforme |
| 18 | Prédécesseurs | `predecessor_states` | oui | oui | Conforme |
| 19 | Sous-automate induit | `induced_subautomaton` | oui (DFA) | oui | Conforme (DFA seulement) |
| 20 | Test de complétude | `is_complete` | oui (DFA) | oui | Conforme (DFA seulement) |
| 21 | Complétion | `complete` | oui (DFA) | oui | Conforme (DFA seulement) |
| 22 | Complément | `complemented` | oui (DFA) | oui | Conforme (DFA seulement) |
| 23 | Produit cartésien | `cartesian_product` | oui (DFA) | oui | Conforme (DFA seulement) |
| 24 | Inclusion | `is_included_in` | oui (DFA) | oui | Conforme (DFA seulement) |
| 25 | Égalité de langages | `has_same_language_as` | oui (DFA) | oui | Conforme (DFA seulement) |
| 26 | Isomorphisme | `is_isomorphic_to` | oui (DFA) | oui | Conforme (DFA seulement) |
| 27 | Préfixe clos | `is_prefix_closed` | oui (DFA) | oui | Conforme |
| 28 | Plus grand sous-langage préfixe clos | `largest_prefix_closed_sublanguage` | oui (DFA) | oui | Conforme |
| 29 | États distinguables | `distinguishing_word`, `are_distinguishable` | oui (DFA) | oui | Conforme |
| 30 | Classes de Myhill-Nerode | `myhill_nerode_classes` | oui (DFA) | oui | Conforme |
| 31 | Minimalité | `is_minimal` | oui (DFA) | oui | Conforme |
| 32 | Minimisation | `minimize` | oui (DFA) | oui | Conforme (DFA partiel reste partiel, documenté) |
| 33 | Déterminisme | `is_deterministic` | oui | oui | Conforme |
| 34 | ε-fermeture | `epsilon_closure`, `epsilon_closure_of` | oui | oui | Conforme |
| 35 | Déterminisation | `determinize` | oui | oui | Conforme |
| 36 | Reverse | `reversed_automaton` | oui | oui | Conforme (nom différent de `reverse` natif) |
| 37 | Suppression des ε | `remove_epsilon_transitions` | oui | oui | Conforme |
| 38 | Langage universel | `is_universal` | oui | oui | Conforme |
| 39 | Élimination d'états (ε-AFEND normalisé) | `normalized`, `eliminate_states` | oui | oui | Partielle : « normalisé » n'est pas défini dans le cahier des charges ; hypothèse documentée |
| 40 | Export DOT | `dot` | oui | oui | Conforme |
| 41 | Export SVG/PDF/TikZ | `svg`, `pdf`, `latex_tikz` | oui | oui | Conforme (SVG vérifié avec Graphviz ; PDF et TikZ compilé dépendent de `dot`/`pdflatex`) |
| 42 | Union | `language_union` | oui | oui | Conforme |
| 43 | Intersection | `language_intersection` | oui | oui | Conforme |
| 44 | Différence | `language_difference` | oui | oui | Conforme |
| 45 | Concaténation | `language_concatenation` | oui | oui | Conforme |
| 46 | Étoile de Kleene | `language_star` | oui | oui | Conforme |
| 47 | Quotient à gauche | `language_left_quotient` | oui | oui | Conforme (ordre des arguments documenté) |
| 48 | Quotient à droite | `language_right_quotient` | oui | oui | Conforme |
| 49 | Dérivées de Brzozowski | `Regex.derivative` | oui | oui | Conforme |
| 50 | Automate de Brzozowski | `Regex.brzozowski_dfa` | oui | oui | Partielle : implémenté comme automate des dérivées ; le cahier des charges cite aussi « double reverse + déterminisation + minimisation », dont la composition n'est pas toujours minimale (voir test d'audit) |
| 51 | Regex → NFA (Thompson) | `Regex.to_nfa` | oui | oui | Conforme |
| 52 | NFA → DFA | `to_dfa` | oui | oui | Conforme |
| 53 | DFA → regex par élimination | `to_regex_elimination` | oui | oui | Partielle : n'utilise pas la chaîne DFA→GNFA→Regex suggérée (pas de GNFA) ; langage vérifié |
| 54 | DFA → regex par Arden | `to_regex_arden` | oui | oui | Conforme |
| 55 | Automate ↔ grammaire régulière | `to_right_grammar`, `to_left_grammar`, `Grammar.to_nfa` | oui | oui | Conforme |
| 56 | Arbres de dérivation | `parse_all`, `ParseTree` | oui | oui | Conforme |
| 57 | Lemme d'Arden | `solve_arden` | oui | oui | Conforme |
| 58 | Systèmes d'équations rationnelles | `to_equation_system`, `EquationSystem` | oui | oui | Conforme |
| 59 | Classes de Myhill-Nerode (mots) | `nerode_partition` | oui | oui | Conforme |
| 60 | Automate résiduel | `residual_automaton` | oui | oui | Conforme |
| 61 | Morphismes | `morphism_to`, `quotient_by`, `minimization_morphism` | oui | oui | Conforme (définitions choisies et documentées dans `morphism.py`) |
| 62 | Monoïde syntaxique | `syntactic_monoid`, `transition_monoid` | oui | oui | Conforme |
