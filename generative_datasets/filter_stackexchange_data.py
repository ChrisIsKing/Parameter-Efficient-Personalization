import pandas as pd

min_num_user_answers = 20
min_num_post_answers = 5

for dataset in ['interpersonal','parenting','travel','workplace','philosophy','worldbuilding','judaism']:#,'health']:
    # users = pd.read_xml(f"data/{dataset}/Users.xml")
    posts = pd.read_xml(f"~/Downloads/{dataset}.stackexchange.com/Posts.xml")

    questions = posts[posts['PostTypeId']==1].drop_duplicates(subset=['Id'])
    answers = posts[posts['PostTypeId']==2].drop_duplicates(subset=['OwnerUserId','ParentId'])

    # filter to posts with min # of answers
    questions = questions[questions['AnswerCount'].astype(int)>=min_num_post_answers]
    answers = answers.groupby('ParentId').filter(lambda x: len(x) >= min_num_post_answers)
    # filter to users with min # of answers
    answers = answers.groupby('OwnerUserId').filter(lambda x: len(x) >= min_num_user_answers)
    # questions = pd.merge(questions.drop('ParentId',axis=1),answers['ParentId'],how='inner',left_on='Id',right_on='ParentId').drop('ParentId',axis=1)

    # print(questions)
    # print(answers)
    # print(len(questions),len(questions[['Id']].drop_duplicates()))
    # print(questions.sort_values('Id').iloc[1],questions.sort_values('Id').iloc[2])
    # print(len(answers), len(answers[['OwnerUserId','ParentId']].drop_duplicates()), len(answers[['Body']].drop_duplicates()))
    print(f"{dataset}: {len(answers)} answers to {len(questions)} questions from {len(answers['OwnerUserId'].unique())} users")
    if len(answers) > 0 and len(questions) > 0:
        # answers.to_xml(f'data/{dataset}.stackexchange.com/Answers.xml')
        # questions.to_xml(f'data/{dataset}.stackexchange.com/Questions.xml')

        merge_df = pd.merge(answers[['OwnerUserId','Body','ParentId']].rename(columns={'OwnerUserId':'UserId','Body':'Answer'}),
                            questions[['Title','Tags','Body','Id']].rename(columns={'Body':'Question','Id':'QuestionId'}),
                            how='left', left_on='ParentId', right_on='QuestionId').drop(columns=['ParentId'])
        merge_df.to_csv(f'data/{dataset}/{dataset}.csv')

# interpersonal: 2228 answers to 1010 questions from 56 users
# parenting: 3908 answers to 1442 questions from 87 users
# travel: 4711 answers to 1973 questions from 92 users
# workplace: 31464 answers to 7105 questions from 435 users
# philosophy: 9819 answers to 2752 questions from 139 users
# worldbuilding: 78910 answers to 12937 questions from 920 users
# judaism: 4918 answers to 1416 questions from 89 users
# health: 0 answers to 20 questions from 0 users