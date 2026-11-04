# Deterministic before/after case review

Two lexicographically first task IDs per category and arm where available. Categories overlap.
Recorded100-term scores are retained; reference checks below cover only the visible20.
A failed-attempt ordinal can differ from the selected portfolio ordinal. Null means no admitted final program.

## 1. static / arm_only_success

Task `sha256:35b36586178bd89092038668ff95e3d9222bcf8da55c28fd6c338bae29d1a135`; inspected attempt 7.

Recorded primary score: success=True, complete=True, reason=success.
Original reference: `{"coverage": 20, "first_values": ["-208", "-29521", "-4048898", "-570724707", "-81042651396"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( local $b0 bigint ) ( local $c0 control ) ( set $b0 ( int -208 ) ) ( loop $c0 ( to_control $index ) ( set $b0 ( mul ( get $b0 ) ( add ( to_bigint ( increment ( get $c0 ) ) ) ( int 138 ) ) ) ) ) ( return ( add ( get $b0 ) ( mul $index ( int -609 ) ) ) ) )
```
Portfolio selected attempt 7; assignment `{"p0": "-983", "p1": "40", "p2": "2"}`.

Transformed:
```text
(program (local $b0 bigint) (local $c0 control) (set $b0 (param.get p0)) (loop $c0 (to_control $index) (set $b0 (mul (get $b0) (add (to_bigint (increment (get $c0))) (param.get p1))))) (return (add (get $b0) (mul $index (param.get p2)))))
```
Final:
```text
(program (local $b0 bigint) (local $c0 control) (set $b0 (int -983)) (loop $c0 (to_control $index) (set $b0 (mul (get $b0) (add (to_bigint (increment (get $c0))) (int 40))))) (return (add (get $b0) (mul $index (int 2)))))
```
Final reference/features: `{"features": {"index_comparisons": [], "literals": 3, "max_literal_digits": 3, "operators": {"add": 2, "get": 3, "increment": 1, "int": 3, "loop": 1, "mul": 2, "set": 2, "to_bigint": 1, "to_control": 1}}, "reference": {"coverage": 20, "first_values": ["-983", "-40301", "-1692722", "-72787212", "-3202637584"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 90 tokens; tail `[')', ')', ')', '(', 'int', '<int>', '1', '3', '8', '</int>', ')', ')', ')', ')', ')', '(', 'return', '(', 'add', '(', 'get', '$b0', ')', '(', 'mul', '$index', '(', 'int', '<int>', '-', '6', '0', '9', '</int>', ')', ')', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [{"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:fa5fb1189096a883b065eef307463204f1c60b94b8d83aefedb48e3f1d86f18b"}], "clock": {"admitted": 2497637.991631857, "completed": 2497638.499637361, "deadline": 2497641.991631857, "dispatched": 2497637.994931812, "owner_observed": 2497638.503991524}, "coefficient_bits": 16, "complete": true, "completed_bounds": [], "job_id": "sha256:b887d7486309f00e30cd65c90ceaa3cdc90be3c92bd2407a5a55934c4b0fd3e6", "outcome": "verified_solution", "phase": "production", "reason": null, "uniqueness": "unique_in_scope"}`.

## 2. static / complete_no_fit

Task `sha256:003fade2ad71bde3165501a3be087c986eb8b267f13f4b544750afc8962e264a`; inspected attempt 0.

Recorded primary score: success=False, complete=True, reason=unsat_in_scope.
Original reference: `{"coverage": 20, "first_values": ["-2020202020200", "-2020202020200", "-2020202020200", "-2020202020200", "-9313"], "reason": null, "status": "ok", "visible_match": null, "visible_values_available": false}`.

Original:
```text
( program ( local $b0 bigint ) ( if ( eq $index ( int 20 ) ) ( then ( set $b0 ( add ( add $index ( int -101 ) ) ( int -1019 ) ) ) ) ( else ( set $b0 ( add ( mul $index ( int -231 ) ) ( int 911 ) ) ) ( if ( le $index ( get $b0 ) ) ( then ( set $b0 ( sub ( int -81313131313131313 ) ( int -81311111111111113 ) ) ) ) ( else ( set $b0 ( int -9313 ) ) ) ) ) ) ( return ( get $b0 ) ) )
```
Final: null (no admitted portfolio selection).
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 178 tokens; tail `['1', '1', '1', '1', '1', '3', '</int>', ')', ')', ')', ')', '(', 'else', '(', 'set', '$b0', '(', 'int', '<int>', '-', '9', '3', '1', '3', '</int>', ')', ')', ')', ')', ')', ')', '(', 'return', '(', 'get', '$b0', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [], "clock": {"admitted": 2497707.726874038, "completed": 2497707.889462127, "deadline": 2497711.726874038, "dispatched": 2497707.727013924, "owner_observed": 2497707.889620607}, "coefficient_bits": 4096, "complete": true, "completed_bounds": [4096], "job_id": "sha256:94d480ed9de423843eeb40c0aa2623d8fb3feed56c79d6c61c6884542276863b", "outcome": "unsat_in_scope", "phase": "affine", "reason": null, "uniqueness": "unchecked"}`.

## 3. static / complete_no_fit

Task `sha256:01fe6576b4317f058decf58e02dbc214a638d6c077e8af74941a424c6bddf719`; inspected attempt 0.

Recorded primary score: success=False, complete=True, reason=unsat_in_scope.
Original reference: `{"coverage": 20, "first_values": ["101", "-519", "-1139", "-1759", "-2379"], "reason": null, "status": "ok", "visible_match": null, "visible_values_available": false}`.

Original:
```text
( program ( return ( sub ( add ( mul $index ( int -619 ) ) ( int -101 ) ) ( add ( add ( int -101 ) $index ) ( int -101 ) ) ) ) )
```
Final: null (no admitted portfolio selection).
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 60 tokens; tail `[')', '(', 'int', '<int>', '-', '1', '0', '1', '</int>', ')', ')', '(', 'add', '(', 'add', '(', 'int', '<int>', '-', '1', '0', '1', '</int>', ')', '$index', ')', '(', 'int', '<int>', '-', '1', '0', '1', '</int>', ')', ')', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [], "clock": {"admitted": 2497581.65633282, "completed": 2497581.814528145, "deadline": 2497585.65633282, "dispatched": 2497581.6574044, "owner_observed": 2497581.817326142}, "coefficient_bits": 4096, "complete": true, "completed_bounds": [4096], "job_id": "sha256:744033d3ace964e9fe9495ba65a95c52f30ca330255e675c49ce550b87e11e5f", "outcome": "unsat_in_scope", "phase": "affine", "reason": null, "uniqueness": "unchecked"}`.

## 4. static / continuation_failure

Task `sha256:060e7ac816d197547a6afbd835565575877e623df6141de0219d411ddd2c0323`; inspected attempt 0.

Recorded primary score: success=False, complete=True, reason=continuation_mismatch.
Original reference: `{"coverage": 20, "first_values": ["0", "-300", "-1200", "-2700", "-4800"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( local $b0 bigint ) ( if ( le $index ( int 20 ) ) ( then ( set $b0 ( mul ( mul $index ( int -300 ) ) $index ) ) ) ( else ( set $b0 ( mul ( add ( mul $index ( int -1000 ) ) ( int -1000 ) ) ( int -10000000000000000000000000000000000000 ) ) ) ) ) ( return ( get $b0 ) ) )
```
Portfolio selected attempt 0; assignment `{"p0": "-315", "p1": "0", "p2": "0", "p3": "0"}`.

Transformed:
```text
(program (local $b0 bigint) (if (le $index (int 20)) (then (set $b0 (mul (mul $index (param.get p0)) $index))) (else (set $b0 (mul (add (mul $index (param.get p1)) (param.get p2)) (param.get p3))))) (return (get $b0)))
```
Final:
```text
(program (local $b0 bigint) (if (le $index (int 20)) (then (set $b0 (mul (mul $index (int -315)) $index))) (else (set $b0 (mul (add (mul $index (int 0)) (int 0)) (int 0))))) (return (get $b0)))
```
Final reference/features: `{"features": {"index_comparisons": [{"operator": "le", "threshold": 20}], "literals": 5, "max_literal_digits": 3, "operators": {"add": 1, "else": 1, "get": 1, "if": 1, "int": 5, "le": 1, "mul": 4, "set": 2, "then": 1}}, "reference": {"coverage": 20, "first_values": ["0", "-315", "-1260", "-2835", "-5040"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 135 tokens; tail `['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '</int>', ')', ')', ')', ')', ')', '(', 'return', '(', 'get', '$b0', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [{"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:79def878d329ecec1ac1342835a701e3c9b25e820a1fa86c4d11cb9284efa9d7"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:4755cddef73711040b9d3d94a59bee507561e7541000cb7c67fe966707adaa46"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:74961a4035400c661c1d34b25b7b04fcd824e61134c9900fb9fd02728acf1bfb"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:13887f25eece0c6ac34d6b45f47adf4ed879c89f67f5c347d7f9308c926d63d7"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:eec604aa246f4364a104179d5368cbe7a20c820248a5f84482d37879a456e04a"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:924086eeb6d2026bfdf02450a183dc882a52d612e620e4b952ab80d1e5b0a6a2"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:f8ddd229ca2c7cf6a53076f817452c9e5ac956cf79e91c40547023f79814c528"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:95fe291432d5d2550f9c8deb9c32dde021812076cf7a43ead57d57c5c290a35d"}], "clock": {"admitted": 2497764.464831797, "completed": 2497764.670449976, "deadline": 2497768.464831797, "dispatched": 2497764.464960172, "owner_observed": 2497764.6725847}, "coefficient_bits": 16, "complete": true, "completed_bounds": [], "job_id": "sha256:92601796c506a28b3a496e7810389c85c6241f078c65ee1739eb5e83cb541769", "outcome": "verified_solution", "phase": "production", "reason": null, "uniqueness": "multiple"}`.

## 5. static / continuation_failure

Task `sha256:0df5dcb23185f1a6da5e16b3eda4f16c6720d7ddadf3ebdd6fda6b022cac705d`; inspected attempt 0.

Recorded primary score: success=False, complete=True, reason=continuation_mismatch.
Original reference: `{"coverage": 20, "first_values": ["200", "210", "220", "230", "240"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( return ( sub ( sub ( mul $index ( int 10 ) ) ( int -100 ) ) ( int -100 ) ) ) )
```
Portfolio selected attempt 0; assignment `{"p0": "0", "p1": "0", "p2": "-475"}`.

Transformed:
```text
(program (return (sub (sub (mul $index (param.get p0)) (param.get p1)) (param.get p2))))
```
Final:
```text
(program (return (sub (sub (mul $index (int 0)) (int 0)) (int -475))))
```
Final reference/features: `{"features": {"index_comparisons": [], "literals": 3, "max_literal_digits": 3, "operators": {"int": 3, "mul": 1, "sub": 2}}, "reference": {"coverage": 20, "first_values": ["475", "475", "475", "475", "475"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 42 tokens; tail `['(', 'return', '(', 'sub', '(', 'sub', '(', 'mul', '$index', '(', 'int', '<int>', '1', '0', '</int>', ')', ')', '(', 'int', '<int>', '-', '1', '0', '0', '</int>', ')', ')', '(', 'int', '<int>', '-', '1', '0', '0', '</int>', ')', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [{"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:0f9fe54444ea7e97063210a3b26e10be3d24509cefe9aa49cef55762bec06b1d"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:df6597d97b8ae7f61cb6029c4dbc27de0fe92f6a51f5c05cd64f50e22b7c51f4"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:277360bfafc11d12e1defcdb96404ca65098b3214350e44cfca5c28cd33c45c7"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:12e1413917fe33f3e990a3500e6ef94bf62cde22083971fb25eae00f98a18fc9"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:ac415d6ba5653d5bf8bff15d32151bb2165b4854858749a48e0f642ce4cc214b"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:e2b5a96ad339f361f712a2c91eaaf3af71ad09a4c62cd0b5ece6712104bed7e9"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:24a4dd3e0d3366c6eb5ea48268b54f79cc281c300e13fdde17e087f4740e49f5"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:45a04901d73aad1b3a9ca85bf51bf196544122abf22ab9381800909c8027def6"}], "clock": {"admitted": 2497581.663678302, "completed": 2497581.890974339, "deadline": 2497585.663678302, "dispatched": 2497581.663789917, "owner_observed": 2497581.891845582}, "coefficient_bits": 16, "complete": true, "completed_bounds": [], "job_id": "sha256:4faacd757cb811e3e3f9d35edda5998282c1036ddd0c972c695a2fdb56a788fd", "outcome": "verified_solution", "phase": "production", "reason": null, "uniqueness": "multiple"}`.

## 6. static / decode_failure

Task `sha256:126e41c5b6864551ab748495ba84beb2117687ee801f097a0c009edd83a7d81f`; inspected attempt 3.

Recorded primary score: success=False, complete=False, reason=grammar_dead_end.
Original reference: `{"reason": "source_missing", "status": "unavailable"}`.

Original:
```text
null
```
Final: null (no admitted portfolio selection).
Generation: `{"complete": false, "outcome": "generation_incomplete", "reason": "grammar_dead_end"}`; 339 tokens; tail `['(', 'then', '(', 'if', '(', 'le', '$index', '(', 'get', '$b0', ')', ')', '(', 'then', '(', 'if', '(', 'le', '$index', '(', 'get', '$b0', ')', ')', '(', 'then', '(', 'if', '(', 'lt', '$index', '(', 'get', '$b0', ')', ')', '(', 'then', '(', 'if']`.
Fit: `{"complete": false, "outcome": "not_attempted", "reason": null}`.

## 7. static / decode_failure

Task `sha256:3d59c54c06ae170038c21dc24be2432a6400a9d932bebe7597877c647899ce8d`; inspected attempt 3.

Recorded primary score: success=True, complete=True, reason=success.
Original reference: `{"reason": "source_missing", "status": "unavailable"}`.

Original:
```text
null
```
Portfolio selected attempt 2; assignment `{"p0": "130", "p1": "-55", "p2": "670"}`.

Transformed:
```text
(program (local $b0 bigint) (if (lt $index (int 2)) (then (set $b0 (mul (mul $index $index) (param.get p0)))) (else (set $b0 (sub (mul (sub $index (param.get p1)) (param.get p2)) $index)))) (return (get $b0)))
```
Final:
```text
(program (local $b0 bigint) (if (lt $index (int 2)) (then (set $b0 (mul (mul $index $index) (int 130)))) (else (set $b0 (sub (mul (sub $index (int -55)) (int 670)) $index)))) (return (get $b0)))
```
Final reference/features: `{"features": {"index_comparisons": [{"operator": "lt", "threshold": 2}], "literals": 4, "max_literal_digits": 3, "operators": {"else": 1, "get": 1, "if": 1, "int": 4, "lt": 1, "mul": 3, "set": 2, "sub": 2, "then": 1}}, "reference": {"coverage": 20, "first_values": ["0", "130", "38188", "38857", "39526"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": false, "outcome": "generation_incomplete", "reason": "grammar_dead_end"}`; 721 tokens; tail `['if', '(', 'lt', '$index', '(', 'int', '<int>', '5', '</int>', ')', ')', '(', 'then', '(', 'if', '(', 'lt', '$index', '(', 'int', '<int>', '6', '2', '0', '9', '</int>', ')', ')', '(', 'then', '(', 'set', '$b0', '$index', ')', ')', '(', 'else', '(', 'if']`.
Fit: `{"complete": false, "outcome": "not_attempted", "reason": null}`.

## 8. static / shared_success

Task `sha256:01915010497616559459510f03b9bbdeeee855b7f1c84a073deadcce6426cbfa`; inspected attempt 0.

Recorded primary score: success=True, complete=True, reason=success.
Original reference: `{"coverage": 20, "first_values": ["-6589", "-5990", "-5391", "-4792", "-4193"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( local $b0 bigint ) ( if ( lt $index ( int 9 ) ) ( then ( set $b0 ( mul ( add $index ( int -11 ) ) ( int 599 ) ) ) ) ( else ( set $b0 ( add ( mul ( mul $index ( int 791 ) ) $index ) $index ) ) ) ) ( return ( get $b0 ) ) )
```
Portfolio selected attempt 0; assignment `{"p0": "-50", "p1": "523", "p2": "780"}`.

Transformed:
```text
(program (local $b0 bigint) (if (lt $index (int 9)) (then (set $b0 (mul (add $index (param.get p0)) (param.get p1)))) (else (set $b0 (add (mul (mul $index (param.get p2)) $index) $index)))) (return (get $b0)))
```
Final:
```text
(program (local $b0 bigint) (if (lt $index (int 9)) (then (set $b0 (mul (add $index (int -50)) (int 523)))) (else (set $b0 (add (mul (mul $index (int 780)) $index) $index)))) (return (get $b0)))
```
Final reference/features: `{"features": {"index_comparisons": [{"operator": "lt", "threshold": 9}], "literals": 4, "max_literal_digits": 3, "operators": {"add": 2, "else": 1, "get": 1, "if": 1, "int": 4, "lt": 1, "mul": 3, "set": 2, "then": 1}}, "reference": {"coverage": 20, "first_values": ["-26150", "-25627", "-25104", "-24581", "-24058"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 86 tokens; tail `[')', ')', ')', '(', 'else', '(', 'set', '$b0', '(', 'add', '(', 'mul', '(', 'mul', '$index', '(', 'int', '<int>', '7', '9', '1', '</int>', ')', ')', '$index', ')', '$index', ')', ')', ')', ')', '(', 'return', '(', 'get', '$b0', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [{"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:2743f45bfc5482f6dd25ebdd5f059bc7b5327fcfe1190cf94b946f43f50fcf07"}], "clock": {"admitted": 2497406.327789586, "completed": 2497406.574203385, "deadline": 2497410.327789586, "dispatched": 2497406.328865563, "owner_observed": 2497406.578985148}, "coefficient_bits": 16, "complete": true, "completed_bounds": [], "job_id": "sha256:224fd1c348cbdb86ce07904630435191fc4674fadb808a19e6a288fc337ae924", "outcome": "verified_solution", "phase": "production", "reason": null, "uniqueness": "unique_in_scope"}`.

## 9. static / shared_success

Task `sha256:0315a16b4ea4e944a95f37983b9479bb8c187d3a5398d5a2b4d9731b97348657`; inspected attempt 0.

Recorded primary score: success=True, complete=True, reason=success.
Original reference: `{"coverage": 20, "first_values": ["0", "-135", "-270", "-405", "-540"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( local $b0 bigint ) ( if ( lt $index ( int 11 ) ) ( then ( set $b0 ( add ( mul $index ( int -136 ) ) $index ) ) ) ( else ( set $b0 ( add ( add ( mul $index ( int 991 ) ) ( int -1010 ) ) $index ) ) ) ) ( return ( get $b0 ) ) )
```
Portfolio selected attempt 0; assignment `{"p0": "-138", "p1": "959", "p2": "309"}`.

Transformed:
```text
(program (local $b0 bigint) (if (lt $index (int 11)) (then (set $b0 (add (mul $index (param.get p0)) $index))) (else (set $b0 (add (add (mul $index (param.get p1)) (param.get p2)) $index)))) (return (get $b0)))
```
Final:
```text
(program (local $b0 bigint) (if (lt $index (int 11)) (then (set $b0 (add (mul $index (int -138)) $index))) (else (set $b0 (add (add (mul $index (int 959)) (int 309)) $index)))) (return (get $b0)))
```
Final reference/features: `{"features": {"index_comparisons": [{"operator": "lt", "threshold": 11}], "literals": 4, "max_literal_digits": 3, "operators": {"add": 3, "else": 1, "get": 1, "if": 1, "int": 4, "lt": 1, "mul": 2, "set": 2, "then": 1}}, "reference": {"coverage": 20, "first_values": ["0", "-137", "-274", "-411", "-548"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 90 tokens; tail `['add', '(', 'add', '(', 'mul', '$index', '(', 'int', '<int>', '9', '9', '1', '</int>', ')', ')', '(', 'int', '<int>', '-', '1', '0', '1', '0', '</int>', ')', ')', '$index', ')', ')', ')', ')', '(', 'return', '(', 'get', '$b0', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [{"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:68d036ea39bb77feb7859df4881b7c90a127538012da7407eb1b39d6fbd03679"}], "clock": {"admitted": 2497184.339547475, "completed": 2497184.518636643, "deadline": 2497188.339547475, "dispatched": 2497184.339701166, "owner_observed": 2497184.521142488}, "coefficient_bits": 4096, "complete": true, "completed_bounds": [], "job_id": "sha256:8ab762a4c94e22d7f0b5b475827f9aac9c7dacf8e694e6d6d90bd628f753441d", "outcome": "verified_solution", "phase": "production", "reason": null, "uniqueness": "unique_in_scope"}`.

## 10. static / template_beats_model

Task `sha256:12c7183a74d7bcb7e5f8d37f9a4313ff39a84df00d9d87eed718bf64e013291d`; inspected attempt 0.

Recorded primary score: success=False, complete=True, reason=unsat_in_scope.
Original reference: `{"coverage": 20, "first_values": ["0", "-2006", "-8024", "-18054", "-32096"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( return ( mul ( mul ( add $index $index ) $index ) ( sub ( add ( int -995 ) ( int -999 ) ) ( int -991 ) ) ) ) )
```
Final: null (no admitted portfolio selection).
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 52 tokens; tail `[')', '$index', ')', '(', 'sub', '(', 'add', '(', 'int', '<int>', '-', '9', '9', '5', '</int>', ')', '(', 'int', '<int>', '-', '9', '9', '9', '</int>', ')', ')', '(', 'int', '<int>', '-', '9', '9', '1', '</int>', ')', ')', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [], "clock": {"admitted": 2497854.048264584, "completed": 2497854.253472513, "deadline": 2497858.048264584, "dispatched": 2497854.048428985, "owner_observed": 2497854.255043647}, "coefficient_bits": 4096, "complete": true, "completed_bounds": [16, 64, 256, 1024, 4096], "job_id": "sha256:59713263881185e7f173f919857af07c65bf549c6a8b66f233abbbc4f3531ef9", "outcome": "unsat_in_scope", "phase": "smt", "reason": null, "uniqueness": "unchecked"}`.

Successful template final:
```text
(program (return (add (mul (int 525441) (mul $index $index)) (int 0))))
```

## 11. static / template_beats_model

Task `sha256:1cfa1b159a0453b5cfe6e13bcb6937c687f22a653e2c02fe613bc95138758f6e`; inspected attempt 0.

Recorded primary score: success=False, complete=True, reason=continuation_mismatch.
Original reference: `{"coverage": 20, "first_values": ["0", "110", "220", "330", "440"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( local $b0 bigint ) ( if ( eq $index ( int 46 ) ) ( then ( set $b0 ( mul ( mul $index ( int -101 ) ) $index ) ) ) ( else ( set $b0 ( mul ( sub ( mul $index ( int -10 ) ) $index ) ( int -10 ) ) ) ) ) ( return ( get $b0 ) ) )
```
Portfolio selected attempt 0; assignment `{"p0": "0", "p1": "462", "p2": "5"}`.

Transformed:
```text
(program (local $b0 bigint) (if (eq $index (int 46)) (then (set $b0 (mul (mul $index (param.get p0)) $index))) (else (set $b0 (mul (sub (mul $index (param.get p1)) $index) (param.get p2))))) (return (get $b0)))
```
Final:
```text
(program (local $b0 bigint) (if (eq $index (int 46)) (then (set $b0 (mul (mul $index (int 0)) $index))) (else (set $b0 (mul (sub (mul $index (int 462)) $index) (int 5))))) (return (get $b0)))
```
Final reference/features: `{"features": {"index_comparisons": [{"operator": "eq", "threshold": 46}], "literals": 4, "max_literal_digits": 3, "operators": {"else": 1, "eq": 1, "get": 1, "if": 1, "int": 4, "mul": 4, "set": 2, "sub": 1, "then": 1}}, "reference": {"coverage": 20, "first_values": ["0", "2305", "4610", "6915", "9220"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 88 tokens; tail `['$b0', '(', 'mul', '(', 'sub', '(', 'mul', '$index', '(', 'int', '<int>', '-', '1', '0', '</int>', ')', ')', '$index', ')', '(', 'int', '<int>', '-', '1', '0', '</int>', ')', ')', ')', ')', ')', '(', 'return', '(', 'get', '$b0', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [{"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:0cab8821bf90fd7868023b86e16b188a3a3793b92705373ce56a58d18dce5b41"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:3db7ce5115b71e64e87ecb5792aa2aeff35c373bb064b962eaaeb923bc377fa8"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:4e3ac1a993d8eb4c1083a2778532f5e8d9d7348f3663091bb40383264be12ee8"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:b7c640671eb061cd2a3ad2ae2f616f1eca538a2b2d874339df951b3211f8713d"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:f3e0152d31253336dad77fdb5926e0fcec7a7249e630b6b11d7cec776ad3ed16"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:9fefa6499c9826746d70e42506a9f55e24fb1e35481fbd091640428dbe948122"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:82933f6d712140d603740a2c0a5dd4c7cffc6096115b93fd603b273da7b32192"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:de53fade54c9f1c11cb3c3b16d2307d04cad4de95d0cb59ae79bf2966fc07521"}], "clock": {"admitted": 2497886.910126267, "completed": 2497887.761673308, "deadline": 2497890.910126267, "dispatched": 2497886.911511331, "owner_observed": 2497887.763685087}, "coefficient_bits": 16, "complete": true, "completed_bounds": [], "job_id": "sha256:01de7ae03f3afb5071d78f13f539633a18f342c6a35f3a38f7dae3a64292bdab", "outcome": "verified_solution", "phase": "production", "reason": null, "uniqueness": "multiple"}`.

Successful template final:
```text
(program (return (add (mul (int 2305) $index) (int 0))))
```

## 12. static / unresolved

Task `sha256:102962cd40a7b131f5614ba0b554b189f8d6f7cbe3f8a2446e1c702165956255`; inspected attempt 0.

Recorded primary score: success=False, complete=False, reason=resource_exhausted:timeout.
Original reference: `{"coverage": 20, "first_values": ["511", "-455790", "404306310", "-359025782280", "318455866217360"], "reason": null, "status": "ok", "visible_match": null, "visible_values_available": false}`.

Original:
```text
( program ( local $b0 bigint ) ( local $c0 control ) ( set $b0 ( int 511 ) ) ( loop $c0 ( to_control $index ) ( set $b0 ( mul ( get $b0 ) ( add ( to_bigint ( increment ( get $c0 ) ) ) ( int -891 ) ) ) ) ) ( return ( add ( get $b0 ) ( mul $index ( int -1000 ) ) ) ) )
```
Final: null (no admitted portfolio selection).
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 91 tokens; tail `[')', '(', 'int', '<int>', '-', '8', '9', '1', '</int>', ')', ')', ')', ')', ')', '(', 'return', '(', 'add', '(', 'get', '$b0', ')', '(', 'mul', '$index', '(', 'int', '<int>', '-', '1', '0', '0', '0', '</int>', ')', ')', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [], "clock": {"admitted": 2497669.738054707, "completed": 2497671.200073542, "deadline": 2497673.738054707, "dispatched": 2497669.738962136, "owner_observed": 2497671.201074633}, "coefficient_bits": 4096, "complete": false, "completed_bounds": [16, 64, 256, 1024], "job_id": "sha256:a0662f8d919a039af74fb2c11d09943f6593e8003ca2c847ca285dde64a3bc40", "outcome": "resource_exhausted", "phase": "smt", "reason": "resource_exhausted:timeout", "uniqueness": "incomplete"}`.

## 13. static / unresolved

Task `sha256:126e41c5b6864551ab748495ba84beb2117687ee801f097a0c009edd83a7d81f`; inspected attempt 3.

Recorded primary score: success=False, complete=False, reason=grammar_dead_end.
Original reference: `{"reason": "source_missing", "status": "unavailable"}`.

Original:
```text
null
```
Final: null (no admitted portfolio selection).
Generation: `{"complete": false, "outcome": "generation_incomplete", "reason": "grammar_dead_end"}`; 339 tokens; tail `['(', 'then', '(', 'if', '(', 'le', '$index', '(', 'get', '$b0', ')', ')', '(', 'then', '(', 'if', '(', 'le', '$index', '(', 'get', '$b0', ')', ')', '(', 'then', '(', 'if', '(', 'lt', '$index', '(', 'get', '$b0', ')', ')', '(', 'then', '(', 'if']`.
Fit: `{"complete": false, "outcome": "not_attempted", "reason": null}`.

## 14. refreshed / arm_only_success

Task `sha256:8ff8f4f176779ecf0d0361291a3b08295985c4f793782d1ac04ae0db6d3b64d6`; inspected attempt 2.

Recorded primary score: success=True, complete=True, reason=success.
Original reference: `{"coverage": 20, "first_values": ["-421456", "-421806", "-422154", "-422500", "-422844"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( return ( mul ( add ( add $index ( int 594 ) ) ( int -97 ) ) ( sub ( add ( int -915 ) $index ) ( int -67 ) ) ) ) )
```
Portfolio selected attempt 2; assignment `{"p0": "560", "p1": "0", "p2": "0", "p3": "0"}`.

Transformed:
```text
(program (return (mul (add (add $index (param.get p0)) (param.get p1)) (sub (add (param.get p2) $index) (param.get p3)))))
```
Final:
```text
(program (return (mul (add (add $index (int 560)) (int 0)) (sub (add (int 0) $index) (int 0)))))
```
Final reference/features: `{"features": {"index_comparisons": [], "literals": 4, "max_literal_digits": 3, "operators": {"add": 3, "int": 4, "mul": 1, "sub": 1}}, "reference": {"coverage": 20, "first_values": ["0", "561", "1124", "1689", "2256"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 57 tokens; tail `['</int>', ')', ')', '(', 'int', '<int>', '-', '9', '7', '</int>', ')', ')', '(', 'sub', '(', 'add', '(', 'int', '<int>', '-', '9', '1', '5', '</int>', ')', '$index', ')', '(', 'int', '<int>', '-', '6', '7', '</int>', ')', ')', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [{"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:2fe23e11401a018ce865195bd000ca2eb61e5f064df14cacdab483620e818ab1"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:4865289b7e69b039acbaf813133dc113eef119b2644b52fe8425907485edb8cd"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:6bb375def87601523161aa6275ae39182faa3bb08381242430724107152fd4dc"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:d893635920ccef5a5cc0b4e17c999dc6cda9e3f6ea9d6215bce66cd7c650d7fd"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:81ae2bdb47fb8e75f3877638b6ed2e14bfb60f2c9d3cab51d2d90f482e8a0623"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:f88a599188ef82f3969a6511d46120e5996049d00f34385ede806f3040f3ec4b"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:55ef7b902a8830def198ec2f6eefad968da52c748353f3cab13035eb2603dde5"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:d5d2c0a3b28329f92b8c7a2b9b216455e9ef2d603cc77bf0dfd99273140d49da"}], "clock": {"admitted": 2501224.467060605, "completed": 2501224.909411539, "deadline": 2501228.467060605, "dispatched": 2501224.470589232, "owner_observed": 2501224.912056192}, "coefficient_bits": 16, "complete": true, "completed_bounds": [], "job_id": "sha256:416dc3b93b3f9c6c72ba27c15f529b7af851eaa45338042dbee85c164d7d24ba", "outcome": "verified_solution", "phase": "production", "reason": null, "uniqueness": "multiple"}`.

## 15. refreshed / complete_no_fit

Task `sha256:01fe6576b4317f058decf58e02dbc214a638d6c077e8af74941a424c6bddf719`; inspected attempt 0.

Recorded primary score: success=False, complete=True, reason=unsat_in_scope.
Original reference: `{"coverage": 20, "first_values": ["187", "-70", "-327", "-584", "-841"], "reason": null, "status": "ok", "visible_match": null, "visible_values_available": false}`.

Original:
```text
( program ( local $b0 bigint ) ( if ( le $index ( int 48 ) ) ( then ( set $b0 ( sub ( mul $index ( int -257 ) ) ( int -187 ) ) ) ) ( else ( set $b0 ( add ( mul ( add $index $index ) ( int -101 ) ) $index ) ) ) ) ( return ( get $b0 ) ) )
```
Final: null (no admitted portfolio selection).
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 90 tokens; tail `[')', ')', '(', 'else', '(', 'set', '$b0', '(', 'add', '(', 'mul', '(', 'add', '$index', '$index', ')', '(', 'int', '<int>', '-', '1', '0', '1', '</int>', ')', ')', '$index', ')', ')', ')', ')', '(', 'return', '(', 'get', '$b0', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [], "clock": {"admitted": 2500892.267208083, "completed": 2500892.422063127, "deadline": 2500896.267208083, "dispatched": 2500892.268117384, "owner_observed": 2500892.422204655}, "coefficient_bits": 4096, "complete": true, "completed_bounds": [4096], "job_id": "sha256:321d706e3bceb942d8f0a23ab9a74ae3fd59aa2eacc262321a833cf713c0e343", "outcome": "unsat_in_scope", "phase": "affine", "reason": null, "uniqueness": "unchecked"}`.

## 16. refreshed / complete_no_fit

Task `sha256:126e41c5b6864551ab748495ba84beb2117687ee801f097a0c009edd83a7d81f`; inspected attempt 0.

Recorded primary score: success=False, complete=True, reason=unsat_in_scope.
Original reference: `{"coverage": 20, "first_values": ["0", "1", "4", "-606", "-1212"], "reason": null, "status": "ok", "visible_match": null, "visible_values_available": false}`.

Original:
```text
( program ( local $b0 bigint ) ( if ( le $index ( int 2 ) ) ( then ( set $b0 ( mul ( mul $index $index ) ( int 1 ) ) ) ) ( else ( set $b0 ( mul ( sub ( mul $index ( int -101 ) ) ( int -101 ) ) $index ) ) ) ) ( return ( get $b0 ) ) )
```
Final: null (no admitted portfolio selection).
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 86 tokens; tail `['mul', '(', 'sub', '(', 'mul', '$index', '(', 'int', '<int>', '-', '1', '0', '1', '</int>', ')', ')', '(', 'int', '<int>', '-', '1', '0', '1', '</int>', ')', ')', '$index', ')', ')', ')', ')', '(', 'return', '(', 'get', '$b0', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [], "clock": {"admitted": 2500841.836333369, "completed": 2500841.993514513, "deadline": 2500845.836333369, "dispatched": 2500841.836439963, "owner_observed": 2500841.99592173}, "coefficient_bits": 4096, "complete": true, "completed_bounds": [4096], "job_id": "sha256:05647189458beb89dea9e1ba18a2af3b8b009e65a8eb03fdfddba875479da99d", "outcome": "unsat_in_scope", "phase": "affine", "reason": null, "uniqueness": "unchecked"}`.

## 17. refreshed / continuation_failure

Task `sha256:060e7ac816d197547a6afbd835565575877e623df6141de0219d411ddd2c0323`; inspected attempt 0.

Recorded primary score: success=False, complete=True, reason=continuation_mismatch.
Original reference: `{"coverage": 20, "first_values": ["0", "-315", "-1260", "-2835", "-5040"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}`.

Original:
```text
( program ( local $b0 bigint ) ( if ( le $index ( int 20 ) ) ( then ( set $b0 ( mul ( mul $index ( int -315 ) ) $index ) ) ) ( else ( set $b0 ( mul ( add ( mul $index ( int -100 ) ) ( int -100 ) ) ( int -100 ) ) ) ) ) ( return ( get $b0 ) ) )
```
Portfolio selected attempt 0; assignment `{"p0": "-315", "p1": "0", "p2": "0", "p3": "0"}`.

Transformed:
```text
(program (local $b0 bigint) (if (le $index (int 20)) (then (set $b0 (mul (mul $index (param.get p0)) $index))) (else (set $b0 (mul (add (mul $index (param.get p1)) (param.get p2)) (param.get p3))))) (return (get $b0)))
```
Final:
```text
(program (local $b0 bigint) (if (le $index (int 20)) (then (set $b0 (mul (mul $index (int -315)) $index))) (else (set $b0 (mul (add (mul $index (int 0)) (int 0)) (int 0))))) (return (get $b0)))
```
Final reference/features: `{"features": {"index_comparisons": [{"operator": "le", "threshold": 20}], "literals": 5, "max_literal_digits": 3, "operators": {"add": 1, "else": 1, "get": 1, "if": 1, "int": 5, "le": 1, "mul": 4, "set": 2, "then": 1}}, "reference": {"coverage": 20, "first_values": ["0", "-315", "-1260", "-2835", "-5040"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 98 tokens; tail `['<int>', '-', '1', '0', '0', '</int>', ')', ')', '(', 'int', '<int>', '-', '1', '0', '0', '</int>', ')', ')', '(', 'int', '<int>', '-', '1', '0', '0', '</int>', ')', ')', ')', ')', ')', '(', 'return', '(', 'get', '$b0', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [{"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:79def878d329ecec1ac1342835a701e3c9b25e820a1fa86c4d11cb9284efa9d7"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:4755cddef73711040b9d3d94a59bee507561e7541000cb7c67fe966707adaa46"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:74961a4035400c661c1d34b25b7b04fcd824e61134c9900fb9fd02728acf1bfb"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:13887f25eece0c6ac34d6b45f47adf4ed879c89f67f5c347d7f9308c926d63d7"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:eec604aa246f4364a104179d5368cbe7a20c820248a5f84482d37879a456e04a"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:924086eeb6d2026bfdf02450a183dc882a52d612e620e4b952ab80d1e5b0a6a2"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:f8ddd229ca2c7cf6a53076f817452c9e5ac956cf79e91c40547023f79814c528"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:95fe291432d5d2550f9c8deb9c32dde021812076cf7a43ead57d57c5c290a35d"}], "clock": {"admitted": 2501054.768776276, "completed": 2501054.974046634, "deadline": 2501058.768776276, "dispatched": 2501054.768936339, "owner_observed": 2501054.976382481}, "coefficient_bits": 16, "complete": true, "completed_bounds": [], "job_id": "sha256:ba9127670aed8b6573706ae555dc8d79461e09ec92b7beb72a871b2adbb8c171", "outcome": "verified_solution", "phase": "production", "reason": null, "uniqueness": "multiple"}`.

## 18. refreshed / continuation_failure

Task `sha256:0df5dcb23185f1a6da5e16b3eda4f16c6720d7ddadf3ebdd6fda6b022cac705d`; inspected attempt 1.

Recorded primary score: success=False, complete=True, reason=continuation_mismatch.
Original reference: `{"coverage": 20, "first_values": ["350", "329", "308", "287", "266"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( return ( sub ( sub ( mul $index ( int -21 ) ) ( int -320 ) ) ( int -30 ) ) ) )
```
Portfolio selected attempt 1; assignment `{"p0": "0", "p1": "0", "p2": "-475"}`.

Transformed:
```text
(program (return (sub (sub (mul $index (param.get p0)) (param.get p1)) (param.get p2))))
```
Final:
```text
(program (return (sub (sub (mul $index (int 0)) (int 0)) (int -475))))
```
Final reference/features: `{"features": {"index_comparisons": [], "literals": 3, "max_literal_digits": 3, "operators": {"int": 3, "mul": 1, "sub": 2}}, "reference": {"coverage": 20, "first_values": ["475", "475", "475", "475", "475"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 42 tokens; tail `['(', 'return', '(', 'sub', '(', 'sub', '(', 'mul', '$index', '(', 'int', '<int>', '-', '2', '1', '</int>', ')', ')', '(', 'int', '<int>', '-', '3', '2', '0', '</int>', ')', ')', '(', 'int', '<int>', '-', '3', '0', '</int>', ')', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [{"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:0f9fe54444ea7e97063210a3b26e10be3d24509cefe9aa49cef55762bec06b1d"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:df6597d97b8ae7f61cb6029c4dbc27de0fe92f6a51f5c05cd64f50e22b7c51f4"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:277360bfafc11d12e1defcdb96404ca65098b3214350e44cfca5c28cd33c45c7"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:12e1413917fe33f3e990a3500e6ef94bf62cde22083971fb25eae00f98a18fc9"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:ac415d6ba5653d5bf8bff15d32151bb2165b4854858749a48e0f642ce4cc214b"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:e2b5a96ad339f361f712a2c91eaaf3af71ad09a4c62cd0b5ece6712104bed7e9"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:24a4dd3e0d3366c6eb5ea48268b54f79cc281c300e13fdde17e087f4740e49f5"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:45a04901d73aad1b3a9ca85bf51bf196544122abf22ab9381800909c8027def6"}], "clock": {"admitted": 2500893.361720253, "completed": 2500893.57455705, "deadline": 2500897.361720253, "dispatched": 2500893.365314048, "owner_observed": 2500893.579017679}, "coefficient_bits": 16, "complete": true, "completed_bounds": [], "job_id": "sha256:9aad954d45ace96100a756d8f252a7c804e0c65caa5c3068dd9ad68e27e6fb3f", "outcome": "verified_solution", "phase": "production", "reason": null, "uniqueness": "multiple"}`.

## 19. refreshed / decode_failure

Task `sha256:4bb1da2f48c7174f3e68db7a5daa37162f1aff33a4f3ad37c0ff370fd65185cf`; inspected attempt 1.

Recorded primary score: success=False, complete=False, reason=token_cap_without_eos.
Original reference: `{"reason": "source_missing", "status": "unavailable"}`.

Original:
```text
null
```
Final: null (no admitted portfolio selection).
Generation: `{"complete": false, "outcome": "generation_incomplete", "reason": "token_cap_without_eos"}`; 1024 tokens; tail `['7', '0', '</int>', ')', ')', '(', 'set', '$b0', '(', 'int', '<int>', '-', '7', '2', '2', '5', '6', '5', '1', '0', '</int>', ')', ')', '(', 'set', '$b0', '(', 'int', '<int>', '-', '2', '</int>', ')', ')', '(', 'set', '$b0', '(', 'int', '<int>']`.
Fit: `{"complete": false, "outcome": "not_attempted", "reason": null}`.

## 20. refreshed / shared_success

Task `sha256:01915010497616559459510f03b9bbdeeee855b7f1c84a073deadcce6426cbfa`; inspected attempt 0.

Recorded primary score: success=True, complete=True, reason=success.
Original reference: `{"coverage": 20, "first_values": ["-6798", "-6180", "-5562", "-4944", "-4326"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( local $b0 bigint ) ( if ( lt $index ( int 9 ) ) ( then ( set $b0 ( mul ( add $index ( int -11 ) ) ( int 618 ) ) ) ) ( else ( set $b0 ( add ( mul ( mul $index ( int 725 ) ) $index ) $index ) ) ) ) ( return ( get $b0 ) ) )
```
Portfolio selected attempt 0; assignment `{"p0": "-50", "p1": "523", "p2": "780"}`.

Transformed:
```text
(program (local $b0 bigint) (if (lt $index (int 9)) (then (set $b0 (mul (add $index (param.get p0)) (param.get p1)))) (else (set $b0 (add (mul (mul $index (param.get p2)) $index) $index)))) (return (get $b0)))
```
Final:
```text
(program (local $b0 bigint) (if (lt $index (int 9)) (then (set $b0 (mul (add $index (int -50)) (int 523)))) (else (set $b0 (add (mul (mul $index (int 780)) $index) $index)))) (return (get $b0)))
```
Final reference/features: `{"features": {"index_comparisons": [{"operator": "lt", "threshold": 9}], "literals": 4, "max_literal_digits": 3, "operators": {"add": 2, "else": 1, "get": 1, "if": 1, "int": 4, "lt": 1, "mul": 3, "set": 2, "then": 1}}, "reference": {"coverage": 20, "first_values": ["-26150", "-25627", "-25104", "-24581", "-24058"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 86 tokens; tail `[')', ')', ')', '(', 'else', '(', 'set', '$b0', '(', 'add', '(', 'mul', '(', 'mul', '$index', '(', 'int', '<int>', '7', '2', '5', '</int>', ')', ')', '$index', ')', '$index', ')', ')', ')', ')', '(', 'return', '(', 'get', '$b0', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [{"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:2743f45bfc5482f6dd25ebdd5f059bc7b5327fcfe1190cf94b946f43f50fcf07"}], "clock": {"admitted": 2500713.983595116, "completed": 2500714.227321123, "deadline": 2500717.983595116, "dispatched": 2500713.983728599, "owner_observed": 2500714.229548311}, "coefficient_bits": 16, "complete": true, "completed_bounds": [], "job_id": "sha256:a2a0b890d1a6accd7e2c348e7b6c8894f07bf48d57477f7b949afac9e8db0ad1", "outcome": "verified_solution", "phase": "production", "reason": null, "uniqueness": "unique_in_scope"}`.

## 21. refreshed / shared_success

Task `sha256:0315a16b4ea4e944a95f37983b9479bb8c187d3a5398d5a2b4d9731b97348657`; inspected attempt 0.

Recorded primary score: success=True, complete=True, reason=success.
Original reference: `{"coverage": 20, "first_values": ["0", "-100", "-200", "-300", "-400"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( local $b0 bigint ) ( if ( lt $index ( int 11 ) ) ( then ( set $b0 ( add ( mul $index ( int -101 ) ) $index ) ) ) ( else ( set $b0 ( add ( add ( mul $index ( int 805 ) ) ( int -1000 ) ) $index ) ) ) ) ( return ( get $b0 ) ) )
```
Portfolio selected attempt 0; assignment `{"p0": "-138", "p1": "959", "p2": "309"}`.

Transformed:
```text
(program (local $b0 bigint) (if (lt $index (int 11)) (then (set $b0 (add (mul $index (param.get p0)) $index))) (else (set $b0 (add (add (mul $index (param.get p1)) (param.get p2)) $index)))) (return (get $b0)))
```
Final:
```text
(program (local $b0 bigint) (if (lt $index (int 11)) (then (set $b0 (add (mul $index (int -138)) $index))) (else (set $b0 (add (add (mul $index (int 959)) (int 309)) $index)))) (return (get $b0)))
```
Final reference/features: `{"features": {"index_comparisons": [{"operator": "lt", "threshold": 11}], "literals": 4, "max_literal_digits": 3, "operators": {"add": 3, "else": 1, "get": 1, "if": 1, "int": 4, "lt": 1, "mul": 2, "set": 2, "then": 1}}, "reference": {"coverage": 20, "first_values": ["0", "-137", "-274", "-411", "-548"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 90 tokens; tail `['add', '(', 'add', '(', 'mul', '$index', '(', 'int', '<int>', '8', '0', '5', '</int>', ')', ')', '(', 'int', '<int>', '-', '1', '0', '0', '0', '</int>', ')', ')', '$index', ')', ')', ')', ')', '(', 'return', '(', 'get', '$b0', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [{"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:68d036ea39bb77feb7859df4881b7c90a127538012da7407eb1b39d6fbd03679"}], "clock": {"admitted": 2500493.229508291, "completed": 2500493.403025648, "deadline": 2500497.229508291, "dispatched": 2500493.22963355, "owner_observed": 2500493.404940124}, "coefficient_bits": 4096, "complete": true, "completed_bounds": [], "job_id": "sha256:382f863a0bbed87979cbda6744aad227d1a4647e434bc465b1ec8d3eb4096941", "outcome": "verified_solution", "phase": "production", "reason": null, "uniqueness": "unique_in_scope"}`.

## 22. refreshed / template_beats_model

Task `sha256:12c7183a74d7bcb7e5f8d37f9a4313ff39a84df00d9d87eed718bf64e013291d`; inspected attempt 0.

Recorded primary score: success=False, complete=True, reason=unsat_in_scope.
Original reference: `{"coverage": 20, "first_values": ["0", "594224", "1188448", "1782672", "2376896"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( local $b0 bigint ) ( if ( eq $index ( int 19 ) ) ( then ( set $b0 ( mul ( sub $index ( int 909 ) ) $index ) ) ) ( else ( set $b0 ( sub ( mul ( mul $index ( int 855 ) ) ( int 695 ) ) $index ) ) ) ) ( return ( get $b0 ) ) )
```
Final: null (no admitted portfolio selection).
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 87 tokens; tail `['$b0', '(', 'sub', '(', 'mul', '(', 'mul', '$index', '(', 'int', '<int>', '8', '5', '5', '</int>', ')', ')', '(', 'int', '<int>', '6', '9', '5', '</int>', ')', ')', '$index', ')', ')', ')', ')', '(', 'return', '(', 'get', '$b0', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [], "clock": {"admitted": 2501148.658090947, "completed": 2501148.865707205, "deadline": 2501152.658090947, "dispatched": 2501148.658430197, "owner_observed": 2501148.870266806}, "coefficient_bits": 4096, "complete": true, "completed_bounds": [16, 64, 256, 1024, 4096], "job_id": "sha256:221ba4620ad1ff096a1a0c43306a66d9f4157ea20b849f26116eb2b177e625a2", "outcome": "unsat_in_scope", "phase": "smt", "reason": null, "uniqueness": "unchecked"}`.

Successful template final:
```text
(program (return (add (mul (int 525441) (mul $index $index)) (int 0))))
```

## 23. refreshed / template_beats_model

Task `sha256:1cfa1b159a0453b5cfe6e13bcb6937c687f22a653e2c02fe613bc95138758f6e`; inspected attempt 0.

Recorded primary score: success=False, complete=True, reason=continuation_mismatch.
Original reference: `{"coverage": 20, "first_values": ["0", "1111", "2222", "3333", "4444"], "reason": null, "status": "ok", "visible_match": false, "visible_values_available": true}`.

Original:
```text
( program ( local $b0 bigint ) ( if ( eq $index ( int 46 ) ) ( then ( set $b0 ( mul ( mul $index ( int -100 ) ) $index ) ) ) ( else ( set $b0 ( mul ( sub ( mul $index ( int -10 ) ) $index ) ( int -101 ) ) ) ) ) ( return ( get $b0 ) ) )
```
Portfolio selected attempt 0; assignment `{"p0": "0", "p1": "462", "p2": "5"}`.

Transformed:
```text
(program (local $b0 bigint) (if (eq $index (int 46)) (then (set $b0 (mul (mul $index (param.get p0)) $index))) (else (set $b0 (mul (sub (mul $index (param.get p1)) $index) (param.get p2))))) (return (get $b0)))
```
Final:
```text
(program (local $b0 bigint) (if (eq $index (int 46)) (then (set $b0 (mul (mul $index (int 0)) $index))) (else (set $b0 (mul (sub (mul $index (int 462)) $index) (int 5))))) (return (get $b0)))
```
Final reference/features: `{"features": {"index_comparisons": [{"operator": "eq", "threshold": 46}], "literals": 4, "max_literal_digits": 3, "operators": {"else": 1, "eq": 1, "get": 1, "if": 1, "int": 4, "mul": 4, "set": 2, "sub": 1, "then": 1}}, "reference": {"coverage": 20, "first_values": ["0", "2305", "4610", "6915", "9220"], "reason": null, "status": "ok", "visible_match": true, "visible_values_available": true}}`.
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 89 tokens; tail `['(', 'mul', '(', 'sub', '(', 'mul', '$index', '(', 'int', '<int>', '-', '1', '0', '</int>', ')', ')', '$index', ')', '(', 'int', '<int>', '-', '1', '0', '1', '</int>', ')', ')', ')', ')', ')', '(', 'return', '(', 'get', '$b0', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [{"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:0cab8821bf90fd7868023b86e16b188a3a3793b92705373ce56a58d18dce5b41"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:3db7ce5115b71e64e87ecb5792aa2aeff35c373bb064b962eaaeb923bc377fa8"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:4e3ac1a993d8eb4c1083a2778532f5e8d9d7348f3663091bb40383264be12ee8"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:b7c640671eb061cd2a3ad2ae2f616f1eca538a2b2d874339df951b3211f8713d"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:f3e0152d31253336dad77fdb5926e0fcec7a7249e630b6b11d7cec776ad3ed16"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:9fefa6499c9826746d70e42506a9f55e24fb1e35481fbd091640428dbe948122"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:82933f6d712140d603740a2c0a5dd4c7cffc6096115b93fd603b273da7b32192"}, {"outcome": "verified_solution", "reason": null, "source_sha256": "sha256:de53fade54c9f1c11cb3c3b16d2307d04cad4de95d0cb59ae79bf2966fc07521"}], "clock": {"admitted": 2501183.399143743, "completed": 2501184.245801048, "deadline": 2501187.399143743, "dispatched": 2501183.399948663, "owner_observed": 2501184.250959193}, "coefficient_bits": 16, "complete": true, "completed_bounds": [], "job_id": "sha256:6fe0d1637ff96f1a911b2210b1097ade5080e084df93bd40bb8d627c6e535453", "outcome": "verified_solution", "phase": "production", "reason": null, "uniqueness": "multiple"}`.

Successful template final:
```text
(program (return (add (mul (int 2305) $index) (int 0))))
```

## 24. refreshed / unresolved

Task `sha256:003fade2ad71bde3165501a3be087c986eb8b267f13f4b544750afc8962e264a`; inspected attempt 5.

Recorded primary score: success=False, complete=False, reason=resource_exhausted:timeout.
Original reference: `{"coverage": 20, "first_values": ["-442232", "-442850", "-443466", "-444080", "-444692"], "reason": null, "status": "ok", "visible_match": null, "visible_values_available": false}`.

Original:
```text
( program ( return ( mul ( add ( add $index ( int 255 ) ) ( int 169 ) ) ( sub ( add ( int -34 ) $index ) ( int 1009 ) ) ) ) )
```
Final: null (no admitted portfolio selection).
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 57 tokens; tail `['</int>', ')', ')', '(', 'int', '<int>', '1', '6', '9', '</int>', ')', ')', '(', 'sub', '(', 'add', '(', 'int', '<int>', '-', '3', '4', '</int>', ')', '$index', ')', '(', 'int', '<int>', '1', '0', '0', '9', '</int>', ')', ')', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [], "clock": {"admitted": 2501006.331611539, "completed": 2501007.791790773, "deadline": 2501010.331611539, "dispatched": 2501006.333064414, "owner_observed": 2501007.792106811}, "coefficient_bits": 16, "complete": false, "completed_bounds": [], "job_id": "sha256:222f84f9fd4a8676f11fea7bcf775278559b56e7519f40499de2b6b5c8ce284a", "outcome": "resource_exhausted", "phase": "smt", "reason": "resource_exhausted:timeout", "uniqueness": "incomplete"}`.

## 25. refreshed / unresolved

Task `sha256:0af9c5e8439b67f3b71fb880c7bbe347654edeb148dfdec35a9063ce5ec06aab`; inspected attempt 0.

Recorded primary score: success=False, complete=False, reason=resource_exhausted:timeout.
Original reference: `{"coverage": 20, "first_values": ["8556", "8357", "8160", "7965", "7772"], "reason": null, "status": "ok", "visible_match": null, "visible_values_available": false}`.

Original:
```text
( program ( return ( mul ( add ( add $index ( int -100 ) ) ( int -38 ) ) ( sub ( add ( int -101 ) $index ) ( int -39 ) ) ) ) )
```
Final: null (no admitted portfolio selection).
Generation: `{"complete": true, "outcome": "decoded", "reason": null}`; 58 tokens; tail `['</int>', ')', ')', '(', 'int', '<int>', '-', '3', '8', '</int>', ')', ')', '(', 'sub', '(', 'add', '(', 'int', '<int>', '-', '1', '0', '1', '</int>', ')', '$index', ')', '(', 'int', '<int>', '-', '3', '9', '</int>', ')', ')', ')', ')', ')', '<eos>']`.
Fit: `{"admission": [], "clock": {"admitted": 2501095.835209675, "completed": 2501097.291953341, "deadline": 2501099.835209675, "dispatched": 2501095.836543822, "owner_observed": 2501097.293458211}, "coefficient_bits": 16, "complete": false, "completed_bounds": [], "job_id": "sha256:9f7b868a8fff1e41caadcec69a8eb1c80854327068d1e9c652233dafea5797aa", "outcome": "resource_exhausted", "phase": "smt", "reason": "resource_exhausted:timeout", "uniqueness": "incomplete"}`.

