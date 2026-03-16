from collections import defaultdict

import graphviz
import numpy as np
import pandas as pd
import pulp
from pulp import LpProblem, LpVariable, LpMinimize, lpSum


def sanitaze_name(name: str) -> str:
    return name.strip().lower().capitalize()


if __name__ == "__main__":
    df = pd.read_csv('Reviews mock data - Sheet1.csv')

    people = df['People'].dropna().apply(sanitaze_name).to_list()
    projects = df['Projects'].dropna().tolist()

    num_people = len(people)
    num_projects = len(projects)

    work_on_project: dict[str, list[str]] = defaultdict(list)
    want_to_review: dict[str, list[str]] = defaultdict(list)

    willing_to_review_two = []

    for _, row in df.iterrows():
        if pd.isna(row['Projects']):
            continue
        for person in row["Project's People"].split(','):
            work_on_project[row['Projects']].append(sanitaze_name(person))
        for person in row["I am fine reviewing this!"].split(','):
            want_to_review[row['Projects']].append(sanitaze_name(person))

    costs = 100 * np.ones((num_people, num_projects))
    for project, people_want_to_review in want_to_review.items():
        for person in people_want_to_review:
            costs[people.index(person), projects.index(project)] -= 1

    edges = [[LpVariable(f"edge_{person}_{project}", cat="Binary") for project in projects] for person in people]

    prob = LpProblem("review_allocation", LpMinimize)
    prob += lpSum([edges[i][j] * costs[i, j] for j in range(num_projects) for i in range(num_people)])
    for person in people:
        prob += lpSum([edges[people.index(person)][j] for j in range(num_projects)]) >= 1

        prob += lpSum([edges[people.index(person)][j] for j in range(num_projects)]) <= 2
        if person not in willing_to_review_two:
            prob += lpSum([edges[people.index(person)][j] for j in range(num_projects)]) == 1

    for project, project_people in work_on_project.items():
        for person in project_people:
            prob += edges[people.index(person)][projects.index(project)] == 0

    for project in projects:
        prob += lpSum([edges[i][projects.index(project)] for i in range(num_people)]) >= 1
        prob += lpSum([edges[i][projects.index(project)] for i in range(num_people)]) <= 2

    status = prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=20))

    adjacency_matrix = np.array([[pulp.value(edges[i][j]) for j in range(len(projects))] for i in range(len(people))])

    print('ALLOCATIONS:')
    for i, person in enumerate(people):
        for j, project in enumerate(projects):
            if adjacency_matrix[i, j]:
                print(f"{person} reviews {project}")

    graph = graphviz.Graph('Project_reviews_allocation', strict=False)

    for project in projects:
        graph.node(project, color='#eedaf5', style='filled')

    for i, person in enumerate(people):
        graph.node(person, color='#daf1f5', style='filled')
        for j, project in enumerate(projects):
            if person in work_on_project[project]:
                graph.edge(person, project, color='black', penwidth='1')
            if adjacency_matrix[i, j]:
                graph.edge(person, project, color='green', penwidth='4', dir='forward')
    graph.view()
