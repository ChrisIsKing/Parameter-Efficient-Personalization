import json
from os.path import join as os_join
from stefutil import *

from peft_u.util.project_paths import *


config_dict = dict(
    datasets=dict(
        cockamanie=dict(
            domain=['humor'],
            instruction="Please rate whether the following text is funny or not funny. "
                        "Respond 'yes' if you think the text is funny and 'no' if you think the text is not funny."
        ),
        gabhate=dict(
            domain=['hate', 'abuse'],
            instruction="Please review the following text and indicate if it has the presence of hate speech. "
                        "Respond 'Hateful' if the text contains hate speech "
                        "and 'Non-hateful' if the text does not contain hate speech."
        ),
        goemotion=dict(
            domain=['emotion'],
            instruction="Please analyze the following text and assign one or more appropriate emotion labels. " 
                        "Emotion labels include happiness, sadness, anger, surprise, joy, fear, disgust. "
                        "You can select one or multiple emotion labels that "
                        "best capture the emotional content of the text. "
                        "Respond with the emotion labels separated by a comma."
        ),
        hatexplain=dict(
            domain=['hate', 'abuse'],
            instruction="Please review the following text "
                        "and indicate if it has the presence of hate speech or is offensive in nature. " 
                        "Respond 'hatespeech' if the text contains hate speech, "
                        "'offensive' if the text does not contain hate speech but is offensive "
                        "and 'normal' if the text is neither offensive nor contains hate speech."
        ),
        measuringhatespeech=dict(
            domain=['hate', 'abuse'],
            instruction="On a scale of 0-2 rate the level of hatespeech contained in the following text. "
                        "Respond with '0' if the text does not contain hate speech, "
                        "'1' if the text contains some hate speech, and '2' if the text contains a lot of hate speech."
        ),
        studemo=dict(
            domain=['emotion'],
            instruction="Please analyze the following text and assign one or more appropriate emotion labels. "
                        "Emotion labels include" 
                        "joy, trust, anticipation, surprise, fear, sadness, disgust, anger, valence, and arousal. "
                        "You can select one or multiple emotion labels "
                        "that best capture the emotional content of the text. "
                        "Respond with the emotion labels separated by a comma."
        ),
        subjectivediscourse_response=dict(
            domain=['discourse'],
            instruction="Please analyze the following text and indicate how the witness responded to the question. "
                        "Respond with 'answer' if they answered the question reasonably, "
                        "'cant-answer-lying' if they could not answer and are lying, "
                        "'can't answer-sincere' if they could not answer but are honest about it, "
                        "'shift-dodge' if they shifted the topic with the intent of dodging the question, "
                        "'answer_overans-sway' if they over answered the question with the intention of swaying "
                        "or 'shift-correct' if they shifted the topic with the intention of clarifying the question."
        ),
        subjectivediscourse_question_sentiment=dict(
            domain=['response sentiment'],
            instruction="Please analyze the following text and rate your sentiment towards the questioners. "
                        "Sentiment labels include 'somewhatPositive', 'positive', 'veryPositive', 'somewhatNegative', "
                        "'veryNegative', 'neutral' and 'negative'. "
                        "Respond with the sentiment label that best captures your sentiment towards the questioners."
        ),
        subjectivediscourse_response_sentiment=dict(
            domain=['response sentiment'],
            instruction="Please analyze the following text and rate your sentiment towards the witness. "
                        "Sentiment labels include 'somewhatPositive', 'positive', 'veryPositive', 'somewhatNegative', "
                        "'veryNegative', 'neutral' and 'negative'. "
                        "Respond with the sentiment label that best captures your sentiment towards the witness."
        ),
        tweeteval=dict(
            domain=['hate', 'abuse'],
            instruction="Please review the following text and indicate if it has the presence of hate speech. "
                        "Respond 'Hateful' if the text contains hate speech "
                        "and 'Non-hateful' if the text does not contain hate speech."
        ),
        unhealthyconversations=dict(
            domain=['hate', 'abuse'],
            instruction="Please review the following text and indicated if it is 'healthy' or 'unhealthy'. "
                        "Respond 'healthy' if the text is healthy and "
                        "'unhealthy' if the text can be considered hostile, antagonistic, condescending, dismissive "
                        "or an unfair generalization."
        ),
        wikidetox=dict(
            domain=['hate', 'abuse'],
            instruction="Please review the following text "
                        "and indicate if it has the presence of malicious remark to a person or group. "
                        "Respond 'Aggressive' if the text contains a personal attack "
                        "and 'Normal' if the text does not contain a personal attack."
        )
    )
)


def add_labels(dataset_name: str = None):
    dset_base_path = os_join(BASE_PATH, PROJ_DIR, DSET_DIR, dataset_name)
    if dataset_name == 'goemotion':
        with open(os_join(dset_base_path, 'ekman_mapping.json')) as f:
            e2detailed_e = json.load(f)
        return list(e2detailed_e.keys()) + ['neutral']


d_dset = config_dict['datasets']
d_dset['goemotion']['labels'] = add_labels('goemotion')


if __name__ == '__main__':
    mic.output_width = 256

    fl_nm = 'config.json'
    mic(config_dict)
    open(fl_nm, 'a').close()  # Create file in OS
    with open(os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', fl_nm), 'w') as f:
        json.dump(config_dict, f, indent=4)
