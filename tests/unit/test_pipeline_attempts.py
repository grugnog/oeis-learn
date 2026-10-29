import json
from pathlib import Path
import tempfile
import unittest
from oeis_learn.pipeline_validation.attempts import AttemptEvidence


class AttemptOutcomes(unittest.TestCase):
    def test_exception_retains_every_scheduled_slot_and_partial_work(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'attempts.jsonl'
            with self.assertRaises(RuntimeError):
                with AttemptEvidence(path) as ledger:
                    ledger.plan(['p'],{'correct':2,'absent':1},parent='contract')
                    ledger.event('p','correct',0,'STARTED',{'seed':17})
                    ledger.event('p','correct',0,'GENERATED',{'tokens':12,'source':'hash'})
                    raise RuntimeError('native timeout')
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            terminal=[row for row in rows if row['state'] in {'INCOMPLETE','UNSTARTED'}]
            self.assertEqual(len(terminal),3)
            self.assertEqual(terminal[0]['state'],'INCOMPLETE')
            self.assertEqual(terminal[0]['charging'],'original reservation retained')
            self.assertEqual(sum(row['state']=='UNSTARTED' for row in terminal),2)

    def test_terminal_and_unknown_slot_rejections(self):
        with tempfile.TemporaryDirectory() as folder:
            with AttemptEvidence(Path(folder)/'attempts.jsonl') as ledger:
                ledger.plan(['p'],{'correct':1},parent='contract')
                ledger.event('p','correct',0,'STARTED',{})
                ledger.event('p','correct',0,'GENERATED',{})
                ledger.event('p','correct',0,'COMMITTED',{})
                with self.assertRaises(ValueError):ledger.event('p','correct',0,'INCOMPLETE',{})
                with self.assertRaises(ValueError):ledger.event('q','correct',0,'STARTED',{})
