import pandas as pd

min_num_user_answers = 20
min_num_post_answers = 5

for dataset in ['interpersonal','parenting','travel','workplace','philosophy','worldbuilding','judaism','health']:
    users = pd.read_xml(f"generative_datasets/{dataset}.stackexchange.com/Users.xml")
    posts = pd.read_xml(f"generative_datasets/{dataset}.stackexchange.com/Posts.xml")

    questions = posts[posts['PostTypeId']==1]
    answers = posts[posts['PostTypeId']==2]

    # filter to posts with min # of answers
    questions = questions[questions['AnswerCount'].astype(int)>=min_num_post_answers]
    answers = answers.groupby('ParentId').filter(lambda x: len(x) >= min_num_post_answers)
    # filter to users with min # of answers
    answers = answers.groupby('OwnerUserId').filter(lambda x: len(x) >= min_num_user_answers)
    questions = pd.merge(questions.drop('ParentId',axis=1),answers['ParentId'],how='left',left_on='Id',right_on='ParentId').drop('ParentId',axis=1)

    # print(questions)
    # print(answers)
    print(f"{dataset}: {len(answers)} answers to {len(questions)} questions from {len(answers['OwnerUserId'].unique())} users")
    if len(answers) > 0 and len(questions) > 0:
        answers.to_xml(f'generative_datasets/{dataset}.stackexchange.com/Answers.xml')
        questions.to_xml(f'generative_datasets/{dataset}.stackexchange.com/Questions.xml')

# interpersonal: 2262 answers to 2343 questions from 57 users
# parenting: 3964 answers to 4034 questions from 88 users
# travel: 4813 answers to 4945 questions from 93 users
# workplace: 31562 answers to 31587 questions from 435 users
# philosophy: 10187 answers to 10238 questions from 145 users
# worldbuilding: 79858 answers to 79868 questions from 926 users
# judaism: 6048 answers to 6057 questions from 99 users
# health: 0 answers to 20 questions from 0 users