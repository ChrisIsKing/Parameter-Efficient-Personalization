import json
from os.path import join as os_join
from stefutil import *

from peft_u.util.project_paths import *


config_dict = dict(
    datasets=dict(
        hatexplain=dict(
            domain=['hate', 'abuse'],
            labels=['hatespeech', 'normal', 'offensive'],
            multi_label=False,
            instruction="Please review the following text "
                        "and indicate if it has the presence of hate speech or is offensive in nature. " 
                        "Respond 'hatespeech' if the text contains hate speech, "
                        "'offensive' if the text does not contain hate speech but is offensive "
                        "and 'normal' if the text is neither offensive nor contains hate speech.",
            is_generative=False
        ),
        gabhate=dict(
            domain=['hate', 'abuse'],
            labels=['Hateful', 'Non-hateful'],
            multi_label=False,
            instruction="Please review the following text and indicate if it has the presence of hate speech. "
                        "Respond 'Hateful' if the text contains hate speech "
                        "and 'Non-hateful' if the text does not contain hate speech.",
            is_generative=False
        ),
        measuringhatespeech=dict(
            domain=['hate', 'abuse'],
            labels=['0', '1', '2'],
            multi_label=False,
            instruction="On a scale of 0-2 rate the level of hatespeech contained in the following text. "
                        "Respond with '0' if the text does not contain hate speech, "
                        "'1' if the text contains some hate speech, and '2' if the text contains a lot of hate speech.",
            is_generative=False
        ),
        tweeteval=dict(
            domain=['hate', 'abuse'],
            labels=['Hateful', 'Non-hateful'],
            multi_label=False,
            instruction="Please review the following text and indicate if it has the presence of hate speech. "
                        "Respond 'Hateful' if the text contains hate speech "
                        "and 'Non-hateful' if the text does not contain hate speech.",
            is_generative=False
        ),
        unhealthyconversations=dict(
            domain=['hate', 'abuse'],
            labels=['healthy', 'unhealthy'],
            multi_label=False,
            instruction="Please review the following text and indicated if it is 'healthy' or 'unhealthy'. "
                        "Respond 'healthy' if the text is healthy and "
                        "'unhealthy' if the text can be considered hostile, antagonistic, condescending, dismissive "
                        "or an unfair generalization.",
            is_generative=False
        ),
        wikidetox=dict(
            domain=['hate', 'abuse'],
            labels=['Aggressive', 'Normal'],
            multi_label=False,
            instruction="Please review the following text "
                        "and indicate if it has the presence of malicious remark to a person or group. "
                        "Respond 'Aggressive' if the text contains a personal attack "
                        "and 'Normal' if the text does not contain a personal attack.",
            is_generative=False
        ),
        goemotion=dict(
            domain=['emotion'],
            labels=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'],
            multi_label=True,
            instruction="Please analyze the following text and assign one or more appropriate emotion labels. " 
                        "Emotion labels include happiness, sadness, anger, surprise, joy, fear, disgust. "
                        "You can select one or multiple emotion labels that "
                        "best capture the emotional content of the text. "
                        "Respond with the emotion labels separated by a comma.",
            is_generative=False
        ),
        studemo=dict(
            domain=['emotion'],
            labels=[
                'anger', 'anticipation', 'arousal', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'valence'
            ],
            multi_label=True,
            instruction="Please analyze the following text and assign one or more appropriate emotion labels. "
                        "Emotion labels include" 
                        "joy, trust, anticipation, surprise, fear, sadness, disgust, anger, valence, and arousal. "
                        "You can select one or multiple emotion labels "
                        "that best capture the emotional content of the text. "
                        "Respond with the emotion labels separated by a comma.",
            is_generative=False
        ),
        cockamamie=dict(
            domain=['humor'],
            labels=['no', 'yes'],
            multi_label=False,
            instruction="Please rate whether the following text is funny or not funny. "
                        "Respond 'yes' if you think the text is funny and 'no' if you think the text is not funny.",
            is_generative=False
        ),
        epic=dict(
            domain=['humor'],
            labels=['Ironic', 'Non-ironic'],
            multi_label=False,
            instruction="Irony is a figurative language device that conveys that opposite of literal meaning, "
                        "profiling intentionally a secondary or extended meaning. "
                        "Please review the following message and reply and indicate if it has the presence of irony. "
                        "Respond 'Ironic' if the reply if you think the reply is ironic "
                        "and 'Non-ironic' if you think the reply is not ironic.",
            is_generative=False
        ),
        subjectivediscourse_response=dict(
            domain=['discourse'],
            labels=[
                'answer', 'answer_overans-sway', 'cant-answer-lying', 'cant-answer-sincere',
                'shift-correct', 'shift-dodge'
            ],
            multi_label=False,
            instruction="Please analyze the following text and indicate how the witness responded to the question. "
                        "Respond with 'answer' if they answered the question reasonably, "
                        "'cant-answer-lying' if they could not answer and are lying, "
                        "'cant-answer-sincere' if they could not answer but are honest about it, "
                        "'shift-dodge' if they shifted the topic with the intent of dodging the question, "
                        "'answer_overans-sway' if they over answered the question with the intention of swaying "
                        "or 'shift-correct' if they shifted the topic with the intention of clarifying the question.",
            is_generative=False
        ),
        subjectivediscourse_question_sentiment=dict(
            domain=['response sentiment'],
            labels=[
                'negative', 'neutral', 'positive',
                'somewhat-negative', 'somewhat-positive', 'very-negative', 'very-positive'
            ],
            multi_label=False,
            instruction="Please analyze the following text and rate your sentiment towards the questioners. "
                        "Sentiment labels include 'somewhat-positive', 'positive', 'very-positive', "
                        "'somewhat-negative', 'very-negative', 'neutral' and 'negative'. "
                        "Respond with the sentiment label that best captures your sentiment towards the questioners.",
            is_generative=False
        ),
        subjectivediscourse_response_sentiment=dict(
            domain=['response sentiment'],
            labels=[
                'negative', 'neutral', 'positive',
                'somewhat-negative', 'somewhat-positive', 'very-negative', 'very-positive'
            ],
            multi_label=False,
            instruction="Please analyze the following text and rate your sentiment towards the witness. "
                        "Sentiment labels include 'somewhat-positive', 'positive', 'very-positive', "
                        "'somewhat-negative', 'very-negative', 'neutral' and 'negative'. "
                        "Respond with the sentiment label that best captures your sentiment towards the witness.",
            is_generative=False
        ),
        goodreads=dict(
            domain=['generative book review response'],
            instruction="Write a book review given the book title and description",
            is_generative=True
        )
        ,
        interpersonal=dict(
            domain=['generative question-answer response'],
            instruction="Answer the question",
            is_generative=True
        )
        ,
        judaism=dict(
            domain=['generative question-answer response'],
            instruction="Answer the question",
            is_generative=True
        )
        ,
        parenting=dict(
            domain=['generative question-answer response'],
            instruction="Answer the question",
            is_generative=True
        )
        ,
        philosophy=dict(
            domain=['generative question-answer response'],
            instruction="Answer the question",
            is_generative=True
        )
        ,
        travel=dict(
            domain=['generative question-answer response'],
            instruction="Answer the question",
            is_generative=True
        )
        ,
        workplace=dict(
            domain=['generative question-answer response'],
            instruction="Answer the question",
            is_generative=True
        )
        ,
        worldbuilding=dict(
            domain=['generative question-answer response'],
            instruction="Answer the question",
            is_generative=True
        )
    )
)


if __name__ == '__main__':
    def run():
        mic.output_width = 256

        fl_nm = 'config.json'
        mic(config_dict)
        open(fl_nm, 'a').close()  # Create file in OS
        with open(os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', fl_nm), 'w') as f:
            json.dump(config_dict, f, indent=4)
    run()
