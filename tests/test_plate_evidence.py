import importlib.util
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import database
from current_plates import CurrentPlates
from plate_evidence import observation, resolve
from plate_validator import is_valid_plate, normalize_plate

FORMATS = [
    ("type1", "A123BC77"),
    ("type1", "A123BC777"),
    ("type1a", "A123BC77"),
    ("type1a", "A123BC777"),
    ("type1b", "AB12377"),
    ("type5", "1234AB77"),
]


def item(frame, text, subtype="type1"):
    return dict(
        frame_id=frame,
        plate=text,
        subtype=subtype,
        confidence=0.96,
        quality=0.8,
    )


class EvidenceTest(unittest.TestCase):
    def test_verified_format_fixtures(self):
        for subtype, text in FORMATS:
            with self.subTest(subtype=subtype, text=text):
                self.assertTrue(is_valid_plate(text, subtype))
                result = resolve([item(i, text, subtype) for i in (1, 2, 3)])
                self.assertEqual(result["status"], "CONFIRMED")
                self.assertEqual(result["subtype"], subtype)
        self.assertFalse(is_valid_plate("AB12377", "type1"))
        self.assertFalse(is_valid_plate("1234AB77", "type1b"))
        self.assertFalse(is_valid_plate("A123BC077", "type1"))
        self.assertFalse(is_valid_plate("", "unknown"))
        self.assertEqual(normalize_plate("А123\nВС 77"), "A123BC77")

    def test_unknown_characters_never_vote(self):
        result = resolve([item(i, "A123##77") for i in range(1, 11)])
        self.assertEqual(result["plate"], "A123##77")
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIsNone(result["confirmed_number"])
        self.assertEqual(result["character_evidence"][4]["frame_ids"], [])

    def test_complementary_partial_frames_can_confirm_observed_symbols(self):
        history = [item(i, "A123##77") for i in (1, 2, 3)]
        history += [item(i, "####BC77") for i in (4, 5, 6)]
        self.assertTrue(all("#" in p["plate"] for p in history))
        result = resolve(history)
        self.assertEqual(result["plate"], "A123BC77")
        self.assertEqual(result["status"], "CONFIRMED")
        self.assertTrue(
            all(len(p["frame_ids"]) >= 3 for p in result["character_evidence"])
        )

    def test_one_clear_read_does_not_fill_missing_confirmation_evidence(self):
        result = resolve(
            [
                item(1, "A123##77"),
                item(2, "A123##77"),
                item(3, "A123##77"),
                item(4, "A123BC77"),
            ]
        )
        self.assertEqual(result["status"], "PENDING")
        self.assertIsNone(result["confirmed_number"])

    def test_conflict_remains_unknown_and_same_frame_cannot_confirm(self):
        history = [
            item(1, "A123BC77"),
            item(2, "A123BC77"),
            item(3, "A123BC77"),
            item(4, "A123BX77"),
        ]
        result = resolve(history)
        self.assertEqual(result["plate"], "A123B#77")
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(
            resolve([item(1, "A123BC77")] * 3)["status"], "PENDING"
        )
        clean = [item(i, "A123BC77") for i in range(5, 15)]
        self.assertEqual(
            resolve((history + clean)[-10:])["status"], "CONFIRMED"
        )

    def test_stable_full_plate_tolerates_isolated_errors_with_quality(self):
        clean = [item(i, "A123BC77") for i in range(1, 9)]
        result = resolve(clean + [item(9, "A123BX77")])
        self.assertEqual(result["plate"], "A123BC77")
        self.assertEqual(result["status"], "CONFIRMED")
        self.assertEqual(len(result["character_evidence"][5]["frame_ids"]), 8)
        # The raw majority is wrong, but its observations have poor quality.
        weak_errors = [
            dict(item(i, "A123BX77"), quality=0.05) for i in range(4, 10)
        ]
        result = resolve(clean[:3] + weak_errors)
        self.assertEqual(result["plate"], "A123BC77")
        self.assertEqual(result["status"], "CONFIRMED")
        # A single good frame must never substitute for three confirmations.
        result = resolve(clean[:1] + weak_errors)
        self.assertNotEqual(result["status"], "CONFIRMED")
        # Use per-character certainty rather than only whole-plate confidence.
        partial = [
            dict(
                item(i, "A123B#77"),
                confidence=0.2,
                character_probabilities={j: 0.98 for j in range(8)},
            )
            for i in (1, 2, 3)
        ]
        self.assertEqual(resolve(partial)["plate"], "A123B#77")

    def test_persistent_ambiguity_masks_only_the_ambiguous_position(self):
        history = [
            item(i, "A123BC77" if i % 2 else "A123BX77") for i in range(1, 11)
        ]
        result = resolve(history)
        self.assertEqual(result["plate"], "A123B#77")
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIsNone(result["confirmed_number"])
        self.assertEqual(result["character_evidence"][5]["frame_ids"], [])
        self.assertTrue(
            all(
                len(p["frame_ids"]) == 10
                for p in result["character_evidence"]
                if p["position"] != 5
            )
        )

    def test_different_tracks_cannot_supply_each_others_missing_symbols(self):
        first = [dict(item(i, "A123##77"), track_id=1) for i in (1, 2, 3)]
        second = [dict(item(i, "####BC77"), track_id=2) for i in (4, 5, 6)]
        self.assertEqual(resolve(first)["plate"], "A123##77")
        self.assertEqual(resolve(second)["plate"], "####BC77")
        with self.assertRaisesRegex(ValueError, "different tracks"):
            resolve(first + second)
        # A newly acquired object starts with independent evidence.
        self.assertEqual(
            resolve([dict(item(7, "B456EK77"), track_id=3)])["status"],
            "PENDING",
        )

    def test_subtypes_and_lengths_never_share_character_votes(self):
        mixed = [item(i, "A123##77", "type1") for i in (1, 2, 3)]
        mixed += [item(i, "####BC77", "type1a") for i in (4, 5, 6)]
        self.assertNotEqual(resolve(mixed)["status"], "CONFIRMED")
        mixed = [item(i, "A123##77") for i in (1, 2, 3)]
        mixed += [item(i, "####BC777") for i in (4, 5, 6)]
        self.assertNotEqual(resolve(mixed)["status"], "CONFIRMED")

    def test_uncertain_ocr_character_is_masked(self):
        plate = dict(
            text="A123BC77",
            subtype="type1",
            readable=True,
            confidence=0.3,
            characters=[
                dict(position=i, probability=0.99 if i != 4 else 0.3)
                for i in range(8)
            ],
        )
        self.assertEqual(observation(plate), ("A123#C77", "type1"))
        plate["subtype_probs"] = [dict(name="type1", probability=0.3)]
        self.assertIsNone(observation(plate))

    def test_legacy_validation_remains_complete_civilian_only(self):
        self.assertTrue(is_valid_plate("A123BC77"))
        self.assertFalse(is_valid_plate("A123##77"))
        self.assertFalse(is_valid_plate("AB12377"))

    def test_sqlite_legacy_migration_status_change_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.db"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute(
                    "CREATE TABLE detections (id INTEGER PRIMARY KEY, "
                    "plate TEXT, confidence REAL, detected_at TEXT)"
                )
                db.execute(
                    "INSERT INTO detections VALUES (1,'A111AA77',.9,'old')"
                )
            with patch.object(database, "DB_PATH", path):
                database.init_db()
                database.init_db()
                legacy = database.get_detections()[0]
                self.assertIsNone(legacy["subtype"])
                self.assertEqual(legacy["status"], "CONFIRMED")
                event = dict(
                    event_id="partial",
                    track_id=1,
                    plate="AB1##77",
                    subtype="type1b",
                    status="PARTIAL",
                    confidence=0.95,
                    quality=0.5,
                )
                self.assertTrue(database.apply_event("s", event))
                self.assertFalse(database.apply_event("s", event))
                event.update(
                    event_id="full", plate="AB12377", status="CONFIRMED"
                )
                self.assertTrue(database.apply_event("s", event))
                rows = database.get_detections()
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["subtype"], "type1b")
                self.assertEqual(rows[0]["status"], "CONFIRMED")

    def test_api_rejects_unknowns_claimed_as_confirmed(self):
        state = CurrentPlates(wall_clock=lambda: 1000)
        with self.assertRaises(ValueError):
            state.update(
                "s",
                [
                    dict(
                        track_id=1,
                        plate="A123##77",
                        subtype="type1",
                        status="CONFIRMED",
                        confidence=0.95,
                        quality=0.5,
                    )
                ],
                lambda *a: True,
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "Torch unavailable"
    )
    def test_ocr_and_runtime_use_the_same_validator(self):
        from ocr_model.utils import alphabet
        from plate_formats import is_valid_plate as shared

        self.assertEqual(alphabet.num_classes, 27)
        self.assertEqual(alphabet.num_subtypes, 30)
        self.assertEqual(alphabet.UNREADABLE_IDX, 3)
        self.assertEqual(alphabet.MAX_SEQ_LEN, 10)
        self.assertIs(alphabet.is_valid_plate, shared)
        for subtype, text in FORMATS:
            self.assertEqual(
                alphabet.is_valid_plate(text, subtype),
                is_valid_plate(text, subtype),
            )


if __name__ == "__main__":
    unittest.main()
