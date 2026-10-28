"""Knowledge/perception and text-only variants of the MLLMU datasets."""

import json
import warnings

from data_process.data_preprocess import (
    LLAVA_multimodal_Dataset,
    collate_multimodal,
    load_image,
    load_metadata,
)

K_TYPE = "Knowledge"
P_TYPE = "Perception"


class MLLMU_manifold_Dataset(LLAVA_multimodal_Dataset):
    def __init__(self, df, full_set, mainfold_type=None, target_size=None, sort_json_key=True):
        # Keep the historical keyword spelling for callers.
        if mainfold_type not in (None, K_TYPE, P_TYPE):
            raise ValueError(f"Unknown manifold type: {mainfold_type}")
        self.full_set = full_set
        self.mainfold_type = mainfold_type
        super().__init__(df, target_size, sort_json_key)

    def get_bio(self, person_id):
        matches = self.full_set.loc[self.full_set["ID"].astype(str) == str(person_id), "biography"]
        if len(matches) != 1:
            raise ValueError(f"Expected one biography for ID {person_id}, found {len(matches)}")
        value = matches.iloc[0]
        try:
            biography = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid biography JSON for ID {person_id}") from error
        if not isinstance(biography, dict):
            raise ValueError(f"Biography for ID {person_id} must be an object")
        return biography

    def flatten_dataset(self):
        flattened_data = []
        for index, row in self.df.iterrows():
            try:
                image = load_image(row["image"])
                metadata = load_metadata(row["metadata"])
            except (OSError, TypeError, ValueError) as error:
                warnings.warn(f"Skipping invalid row {index}: {error}", stacklevel=2)
                continue
            for qa_pair in metadata:
                question, answer = qa_pair.get("Question", ""), qa_pair.get("Answer", "")
                person_id = qa_pair.get("ID")
                if person_id is None or person_id == "":
                    continue
                biography = self.get_bio(person_id)
                if self.mainfold_type == P_TYPE:
                    if "name" in question:
                        appearance = biography["appearance_detail"]
                        question, answer = appearance["question"], appearance["answer"]
                    elif "gender" not in question:
                        continue
                if question and answer:
                    flattened_data.append(
                        {
                            "image": image,
                            "question": question,
                            "answer": answer,
                            "person_ID": person_id,
                            "biography": biography,
                        }
                    )
        return flattened_data

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = self.resize_image(sample["image"])
        question = sample["question"]
        if self.mainfold_type == K_TYPE:
            pronouns = ("this person", "the person", "this individual", "the individual")
            if any(pronoun in question for pronoun in pronouns):
                image = None
                for pronoun in pronouns:
                    question = question.replace(pronoun, sample["biography"]["Name"])
        return {
            "image": image,
            "question": self.json2token(question, self.sort_json_key),
            "answer": self.json2token(sample["answer"], self.sort_json_key),
        }


class MLLMU_text_Dataset(LLAVA_multimodal_Dataset):
    def flatten_dataset(self):
        flattened_data = []
        for _, row in self.df.iterrows():
            questions = row["Classification_Task"]
            if isinstance(questions, str):
                questions = json.loads(questions)
            for qa in questions.get("Pure_Text_Questions", []):
                options, correct = qa["Options"], qa["Correct_Answer"]
                if isinstance(options, dict):
                    answer = options[correct]
                elif isinstance(correct, str) and len(correct) == 1 and correct.isalpha():
                    option_index = ord(correct.upper()) - ord("A")
                    if not 0 <= option_index < len(options):
                        raise ValueError(f"Invalid option label: {correct}")
                    answer = options[option_index]
                else:
                    answer = options[correct]
                flattened_data.append({"image": None, "question": qa["Question"], "answer": answer})
        return flattened_data


def train_collate_fn_llava_new(examples, processor, train_flag=True, max_length=None):
    return collate_multimodal(examples, processor, train_flag=train_flag, max_length=max_length)
