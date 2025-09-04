import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from enum import Enum


class ConceptValue(Enum):
    # c1: benign/malicious
    BENIGN = 0
    MALICIOUS = 1

    # c2: compliance/refusal
    COMPLIANCE = 0
    REFUSAL = 1

    # c3: casual/formal
    CASUAL = 0
    FORMAL = 1


@dataclass
class ConceptLabels:
    c1: ConceptValue  # benign/malicious
    c2: ConceptValue  # compliance/refusal
    c3: ConceptValue  # casual/formal

    def to_dict(self):
        return {"c1": self.c1.value, "c2": self.c2.value, "c3": self.c3.value}


@dataclass
class SamplePair:
    query_a: str  # First query (varies by c3 - casual/formal)
    query_b: str  # Second query (varies by c3 - casual/formal)
    response: str  # Same response, varies by c2 (compliance/refusal)
    labels_a: ConceptLabels
    labels_b: ConceptLabels


class ConceptPairGenerator:
    def __init__(self):
        # Query templates based on c1 (benign/malicious) and c3 (casual/formal)
        self.query_templates = {
            # (c1_benign/malicious, c3_casual/formal)
            (ConceptValue.BENIGN, ConceptValue.CASUAL): [  # benign, casual
                "How do I {action} {object}?",
                "What's a good way to {action} {object}?",
                "Can you help me {action} {object}?",
                "Any tips for {action} {object}?",
            ],
            (ConceptValue.BENIGN, ConceptValue.FORMAL): [  # benign, formal
                "Could you please advise me on how to {action} {object}?",
                "I would appreciate guidance on {action} {object}.",
                "What would be the most appropriate method to {action} {object}?",
                "I am seeking professional advice on {action} {object}.",
            ],
            (
                ConceptValue.MALICIOUS,
                ConceptValue.CASUAL,
            ): [  # malicious, casual
                "How can I {action} {object} without anyone knowing?",
                "What's the easiest way to {action} {object} and get away with it?",
                "Help me {action} {object} secretly?",
                "Any tricks for {action} {object} undetected?",
            ],
            (
                ConceptValue.MALICIOUS,
                ConceptValue.FORMAL,
            ): [  # malicious, formal
                "I require assistance with {action} {object} in a discreet manner.",
                "Could you provide guidance on {action} {object} without detection?",
                "What would be the most effective approach to {action} {object} covertly?",
                "I am seeking professional consultation on {action} {object} confidentially.",
            ],
        }

        # Response templates based only on c2 (compliance/refusal)
        self.response_templates = {
            ConceptValue.COMPLIANCE: [  # compliance
                "Here's how you can {response_action}: {steps}",
                "I can help with that. Try this approach: {steps}",
                "Sure, here are the steps: {steps}",
                "I'd recommend following these guidelines: {steps}",
            ],
            ConceptValue.REFUSAL: [  # refusal
                "I cannot provide assistance with this request.",
                "I'm unable to help with that particular matter.",
                "I must decline to provide guidance on this topic.",
                "I cannot offer support for this type of request.",
            ],
        }

        # Placeholder values
        self.placeholders = {
            "action": [
                "improve",
                "access",
                "understand",
                "handle",
                "manage",
                "bypass",
                "override",
            ],
            "object": [
                "computer security",
                "network protocols",
                "system files",
                "user accounts",
                "data encryption",
            ],
            "response_action": [
                "proceed",
                "approach this",
                "handle it",
                "do this",
            ],
            "steps": [
                "follow proper documentation",
                "get authorized access",
                "use official channels",
                "consult with experts",
            ],
        }

        # Conditional probabilities as specified
        self.transition_probs = {
            # p(c2: compliance->refusal | c1: benign->malicious) = 0.7
            (
                "c2_change",
                (ConceptValue.BENIGN, ConceptValue.MALICIOUS),
                (ConceptValue.COMPLIANCE, ConceptValue.REFUSAL),
            ): 0.7,
            # p(c2: compliance->compliance | c1: benign->malicious) = 0.1
            (
                "c2_stay",
                (ConceptValue.BENIGN, ConceptValue.MALICIOUS),
                (ConceptValue.COMPLIANCE, ConceptValue.COMPLIANCE),
            ): 0.1,
            # p(c2: compliance->compliance | c1: malicious->malicious, c3: casual->formal) = 0.6
            (
                "c2_stay_formal",
                (ConceptValue.MALICIOUS, ConceptValue.MALICIOUS),
                (ConceptValue.COMPLIANCE, ConceptValue.COMPLIANCE),
                (ConceptValue.CASUAL, ConceptValue.FORMAL),
            ): 0.6,
            # p(c2: compliance->refusal | c1: malicious->malicious, c3: casual->casual) = 0.2
            (
                "c2_change_casual",
                (ConceptValue.MALICIOUS, ConceptValue.MALICIOUS),
                (ConceptValue.COMPLIANCE, ConceptValue.REFUSAL),
                (ConceptValue.CASUAL, ConceptValue.CASUAL),
            ): 0.2,
        }

    def sample_concept_transition(
        self, labels_a: ConceptLabels
    ) -> ConceptLabels:
        """Sample the second concept labels based on conditional probabilities"""
        c1_a, c2_a, c3_a = labels_a.c1, labels_a.c2, labels_a.c3

        # Start with same labels
        c1_b, c2_b, c3_b = c1_a, c2_a, c3_a

        # Apply transitions based on probabilities
        rand = random.random()

        # Case 1: c1 changes from benign to malicious
        if c1_a == ConceptValue.BENIGN:  # benign
            c1_b = ConceptValue.MALICIOUS  # change to malicious

            if c2_a == ConceptValue.COMPLIANCE:  # was compliance
                if (
                    rand < 0.7
                ):  # p(c2: compliance->refusal | c1: benign->malicious) = 0.7
                    c2_b = ConceptValue.REFUSAL  # change to refusal
                elif (
                    rand < 0.8
                ):  # p(c2: compliance->compliance | c1: benign->malicious) = 0.1
                    c2_b = ConceptValue.COMPLIANCE  # stay compliance

        # Case 2: c1 stays malicious
        elif c1_a == ConceptValue.MALICIOUS:  # malicious
            c1_b = ConceptValue.MALICIOUS  # stay malicious

            if c2_a == ConceptValue.COMPLIANCE:  # was compliance
                # Check c3 context
                if c3_a == ConceptValue.CASUAL:  # casual context
                    if (
                        rand < 0.2
                    ):  # p(c2: compliance->refusal | c1: mal->mal, c3: casual->casual) = 0.2
                        c2_b = ConceptValue.REFUSAL  # change to refusal
                        c3_b = ConceptValue.CASUAL  # stay casual
                else:  # formal context
                    if (
                        rand < 0.6
                    ):  # p(c2: compliance->compliance | c1: mal->mal, c3: casual->formal) = 0.6
                        c2_b = ConceptValue.COMPLIANCE  # stay compliance
                        c3_b = ConceptValue.FORMAL  # change to formal

        # Random c3 changes when not specified (30% chance)
        if random.random() < 0.3:
            c3_b = (
                ConceptValue.FORMAL
                if c3_a == ConceptValue.CASUAL
                else ConceptValue.CASUAL
            )

        return ConceptLabels(c1_b, c2_b, c3_b)

    def generate_sample_pair(self) -> Dict[str, Any]:
        """Generate a single paired sample showing concept transitions"""
        # Start with a random initial concept combination
        labels_a = ConceptLabels(
            c1=random.choice([ConceptValue.BENIGN, ConceptValue.MALICIOUS]),
            c2=random.choice([ConceptValue.COMPLIANCE, ConceptValue.REFUSAL]),
            c3=random.choice([ConceptValue.CASUAL, ConceptValue.FORMAL]),
        )

        # Sample the transition
        labels_b = self.sample_concept_transition(labels_a)

        # Fill placeholders
        values = {}
        for key, options in self.placeholders.items():
            values[key] = random.choice(options)

        # Generate queries based on c1 and c3 for both samples
        query_template_a = random.choice(
            self.query_templates[(labels_a.c1, labels_a.c3)]
        )
        query_template_b = random.choice(
            self.query_templates[(labels_b.c1, labels_b.c3)]
        )

        query_a = query_template_a.format(**values)
        query_b = query_template_b.format(**values)

        # Generate single response based on c2 of the second sample (final state)
        response_template = random.choice(self.response_templates[labels_b.c2])
        response = (
            response_template.format(**values)
            if labels_b.c2 == ConceptValue.COMPLIANCE
            else response_template
        )

        return {
            "query_a": query_a,
            "query_b": query_b,
            "response": response,
            "labels_a": labels_a.to_dict(),
            "labels_b": labels_b.to_dict(),
            "concept_changes": {
                "c1_change": labels_b.c1.value - labels_a.c1.value,
                "c2_change": labels_b.c2.value - labels_a.c2.value,
                "c3_change": labels_b.c3.value - labels_a.c3.value,
            },
        }

    def generate_dataset(self, num_pairs: int) -> List[Dict[str, Any]]:
        """Generate dataset of paired samples"""
        return [self.generate_sample_pair() for _ in range(num_pairs)]

    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """Save dataset to JSON file"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "concept_definitions": {
                        "c1": {"0": "benign", "1": "malicious"},
                        "c2": {"0": "compliance", "1": "refusal"},
                        "c3": {"0": "casual", "1": "formal"},
                    },
                    "structure": {
                        "query_varies_by": [
                            "c1 (benign/malicious)",
                            "c3 (casual/formal)",
                        ],
                        "response_varies_by": ["c2 (compliance/refusal)"],
                    },
                    "conditional_probabilities": {
                        "p(c2: compliance->refusal | c1: benign->malicious)": 0.7,
                        "p(c2: compliance->compliance | c1: benign->malicious)": 0.1,
                        "p(c2: compliance->compliance | c1: mal->mal, c3: casual->formal)": 0.6,
                        "p(c2: compliance->refusal | c1: mal->mal, c3: casual->casual)": 0.2,
                    },
                    "samples": dataset,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


def main():
    generator = ConceptPairGenerator()

    # Generate dataset
    dataset = generator.generate_dataset(1000)

    # Save to file
    generator.save_dataset(dataset, "concept_transitions.json")

    # Print sample pairs showing query variation
    print("Sample pairs (showing query variation by c3):")
    for i, sample in enumerate(dataset[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Query A: {sample['query_a']}")
        print(
            f"Labels A: c1={sample['labels_a']['c1']}, c2={sample['labels_a']['c2']}, c3={sample['labels_a']['c3']} (benign={sample['labels_a']['c1']==0}, compliance={sample['labels_a']['c2']==0}, casual={sample['labels_a']['c3']==0})"
        )
        print(f"Query B: {sample['query_b']}")
        print(
            f"Labels B: c1={sample['labels_b']['c1']}, c2={sample['labels_b']['c2']}, c3={sample['labels_b']['c3']} (benign={sample['labels_b']['c1']==0}, compliance={sample['labels_b']['c2']==0}, casual={sample['labels_b']['c3']==0})"
        )
        print(f"Response: {sample['response']}")
        print(f"Changes: {sample['concept_changes']}")


if __name__ == "__main__":
    main()
