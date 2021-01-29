import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


Churners = pd.read_csv("BankChurners.csv")
sns.set(rc={'figure.figsize': (10, 5)})

column_types = {"numerical": ["Customer_Age", "Dependent_count", "Months_on_book", "Total_Relationship_Count",
                              "Months_Inactive_12_mon",
                              "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy",
                              "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1",
                              "Avg_Utilization_Ratio",
                              "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
                              "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"],

                "ordinal": ["Education_Level", "Income_Category"],

                "categorical": ["Attrition_Flag", "Gender", "Marital_Status", "Card_Category"]}


def visualize_categorical_dependence():
    for cat in (column_types["categorical"] + column_types["ordinal"]):
        sns.countplot(x=Churners[cat], order=list(set(Churners[cat].array)))
        plt.show()

        for num in column_types["numerical"]:
            print(cat + " to " + num + " dependence")
            if num == "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1" \
                    or num == "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2":
                sns.boxplot(x=Churners[cat], y=Churners[num], order=list(set(Churners[cat].array)), data=Churners)
            else:
                sns.violinplot(x=Churners[cat], y=Churners[num], order=list(set(Churners[cat].array)), data=Churners)

            plt.show()


def visualize_numerical_dependence():
    for i in range(len(column_types["numerical"])):
        for j in range(i, len(column_types["numerical"])):
            num1 = column_types["numerical"][i]
            num2 = column_types["numerical"][j]
            if num1 == num2:
                continue
            else:
                if num1 in ["Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Dependent_count"] \
                  or num2 in ["Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Dependent_count"]:
                    print(num1 + " to " + num2 + " dependence")
                    sns.displot(Churners, x=num1, y=num2, kind="kde")
                else:
                    print(num1 + " to " + num2 + " dependence")
                    sns.displot(Churners, x=num1, y=num2)

            plt.show()


def count_frequency():
    print("=======================FREQUENCY==========================")
    for l in column_types.values():
        for series in l:
            print(Churners[series].value_counts(normalize=True, sort=True), "\n")

    print("\n")


def count_mean():
    print("=========================MEAN=============================")
    for series in column_types["numerical"]:
        print(series + " mean: ", Churners[series].mean())

    print("\n")


def count_std_deviation():
    print("=======================DEVIATION==========================")
    for series in column_types["numerical"]:
        print(series + " standart deviation: ", Churners[series].std())
    print("\n")


def build_corr_matrix():
    df = pd.DataFrame(Churners, columns=column_types["numerical"])
    print(df.corr())


count_mean()
count_std_deviation()
count_frequency()
build_corr_matrix()
visualize_categorical_dependence()
visualize_numerical_dependence()
