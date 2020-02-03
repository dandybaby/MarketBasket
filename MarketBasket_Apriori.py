import pandas as pd
import time

# 数据加载
from fptools import fpgrowth

data = pd.read_csv('./Market_Basket_Optimisation.csv', header=None)
data.fillna(0, inplace=True)

# 数据预处理
transactions = []
for i in range(0, len(data)):
    transactions.append([str(data.values[i, j])
                         for j in range(0, data.shape[1]) if str(data.values[i, j]) != '0'])


# 方法1
# 采用efficient_apriori工具包
def rule1():
    from efficient_apriori import apriori
    start = time.time()
    # 挖掘频繁项集和频繁规则
    itemsets, rules = apriori(transactions, min_support=0.03, min_confidence=0.2)
    print('频繁项集：', itemsets)
    df_results1 = pd.DataFrame(rules)
    print('关联规则：', df_results1)
    end = time.time()
    print("用时：", end - start)


# 方法2
# 采用mlxtend.frequent_patterns工具包
def rule2():
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    from mlxtend.preprocessing import TransactionEncoder
    start = time.time()
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.03, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2)
    print("频繁项集：", frequent_itemsets)
    print("关联规则：", rules[(rules['lift'] >= 1) & (rules['confidence'] >= 0.2)])
    # print(rules['confidence'])
    end = time.time()
    print("用时：", end - start)


def rule3():
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.frequent_patterns import association_rules
    from mlxtend.preprocessing import TransactionEncoder
    now = time.time()
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = fpgrowth(df, min_support=0.03, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2)
    print("频繁项集：", frequent_itemsets)
    print("关联规则：", rules[(rules['lift'] >= 1) & (rules['confidence'] >= 0.2)])
    print("用时：", time.time() - now)


rule1()
print('-' * 100)
rule2()
print('-' * 100)
rule3()
