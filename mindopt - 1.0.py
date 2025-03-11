"""函数库"""
import pandas as pd
import datetime
from gurobipy import Model
from gurobipy import GRB
import gurobipy as gp
# 读取 Excel 文件
file_path = 'plan111.xlsx'
PLAN = pd.read_excel(file_path, header=None)

"""函数"""
# 时间转换函数(datatime转换为分钟数)
def time_to_minutes(time_obj):
    return time_obj.hour * 60 + time_obj.minute

#时间转换函数(分钟数转化为datatime)
def minutes_to_time(minutes):
    # 计算小时数
    hours = int(minutes // 60)
    # 计算剩余的分钟数
    remaining_minutes = int(minutes % 60)
    # 创建并返回时间对象
    return datetime.time(hours, remaining_minutes)
"""参数定义"""
time_delay = 800#晚点时间
M = 1000000#一个极大值
h_da = 2#到发时间间隔
h_dd = h_aa = 5#到到时间间隔、发发时间间隔
s = 1#最小停车时间





"""数据整理"""
rows, cols = PLAN.shape
num_station = rows - 1
num_train = int((cols - 3)/3)
train_name = PLAN.iloc[0, list(range(3, PLAN.shape[1], 3))].reset_index(drop=True)
train_name = train_name.str.split('_').str[0]#列车信息
station = PLAN.iloc[1:, 2].reset_index(drop=True)
train_arrive = PLAN.iloc[1:, list(range(3, PLAN.shape[1], 3))].reset_index(drop=True)
train_arrive = train_arrive.fillna('0:0:0')
train_arrive = train_arrive.applymap(lambda x: pd.to_datetime(x, format='%H:%M:%S').time())
train_arrive = train_arrive.applymap(time_to_minutes)
train_departure = PLAN.iloc[1:, list(range(4, PLAN.shape[1], 3))].reset_index(drop=True)
train_departure = train_departure.fillna('0:0:0')
train_departure = train_departure.applymap(lambda x: pd.to_datetime(x, format='%H:%M:%S').time())
train_departure = train_departure.applymap(time_to_minutes)
print("ok")


"""高铁列车调度优化模型"""

#创建模型
mindopt_gurobi = gp.Model("mindopt")

# 允许的目标函数相对误差为 1%
MYMIPGAP = 0.01
mindopt_gurobi.Params.MIPGap = MYMIPGAP

#定义模型的决策变量
time_arrive = mindopt_gurobi.addVars(num_station, num_train, lb=0, name=" time_arrive", vtype=GRB.CONTINUOUS)#实际到站时间
time_departure = mindopt_gurobi.addVars(num_station, num_train, lb=0, name=" time_departure", vtype=GRB.CONTINUOUS)#实际发车时间
train_stop = mindopt_gurobi.addVars(num_station, num_train, lb=0, name=" train_stop", vtype=GRB.BINARY)#是否停车
train_order = mindopt_gurobi.addVars(num_station - 1, num_train, num_train, lb=0, name=" train_order", vtype=GRB.BINARY)#区间列车发车顺序


#目标函数
mindopt_gurobi.setObjective(
    gp.quicksum(time_arrive[i, k] - train_arrive.iloc[i, k] for i in range(num_station) for k in range(num_train)) +
    gp.quicksum(time_departure[i, k] - train_departure.iloc[i, k] for i in range(num_station) for k in range(num_train))
)

#约束条件
#约束一：实际到发时间不得早于计划到发时间
for i in range(num_station):
    for k in range(num_train):
        mindopt_gurobi.addConstr(time_arrive[i, k] >= train_arrive.iloc[i, k])
        mindopt_gurobi.addConstr(time_departure[i, k] >= train_departure.iloc[i, k])

#约束二：在晚点发生之前，实际到发时间等于计划到发时间
for i in range(num_station):
    for k in range(num_train):
        if train_arrive.iloc[i, k] <= time_delay:
            mindopt_gurobi.addConstr(time_arrive[i, k] == train_arrive.iloc[i, k])
        if train_departure.iloc[i, k] <= time_delay:
            mindopt_gurobi.addConstr(time_departure[i, k] == train_departure.iloc[i, k])
#约束三：到发时间约束
for i in range(num_station):
    for k in range(num_train):
        mindopt_gurobi.addConstr(time_arrive[i, k] <= time_departure[i, k])
for i in range(num_station - 1):
    for k in range(num_train):
        mindopt_gurobi.addConstr(time_departure[i, k] <= time_arrive[i + 1, k])
#约束四：区间禁止越行约束
for i in range(num_station - 1):
    for k in range(num_train):
        for j in range(num_train):
            if k != j:
                mindopt_gurobi.addConstr(time_departure[i, j] >= time_departure[i, k] - (1 - train_order[i, k, j]) * M)
                mindopt_gurobi.addConstr(time_departure[i, k] >= time_departure[i, j] - train_order[i, k, j] * M)
                mindopt_gurobi.addConstr(time_arrive[i + 1, j] >= time_arrive[i + 1, k] - (1 - train_order[i, k, j]) * M)
                mindopt_gurobi.addConstr(time_arrive[i + 1, k] >= time_arrive[i + 1, j] - train_order[i, k, j] * M)
# #约束五：最小时间间隔约束
# #到到时间间隔约束
# for i in range(1, num_station):
#     for k in range(num_train):
#         for j in range(num_train):
#             if k != j:
#                 mindopt_gurobi.addConstr(time_arrive[i, j] >= time_arrive[i, k] + h_aa - (1 - train_order[i - 1, k, j]) * M)
#                 mindopt_gurobi.addConstr(time_arrive[i, k] >= time_arrive[i, j] + h_aa - train_order[i - 1, k, j] * M)
# #发发时间间隔约束
# for i in range(num_station - 1):
#     for k in range(num_train):
#         for j in range(num_train):
#             if k != j:
#                 mindopt_gurobi.addConstr(time_departure[i, j] >= time_departure[i, k] + h_dd - (1 - train_order[i, k, j]) * M)
#                 mindopt_gurobi.addConstr(time_departure[i, k] >= time_departure[i, j] + h_dd - train_order[i, k, j] * M)
# #发到时间间隔约束
# for i in range(1, num_station - 1):
#     for k in range(num_train):
#         for j in range(num_train):
#             if k != j:
#                 mindopt_gurobi.addConstr(time_arrive[i, j] >= time_departure[i, k] + h_da - (1 - train_order[i - 1, k, j]) * M)
#                 mindopt_gurobi.addConstr(time_arrive[i, k] >= time_departure[i, j] + h_da - train_order[i - 1, k, j] * M)
#约束六：最小停车时间
for i in range(num_station):
    for k in range(num_train):
        if train_arrive.iloc[i, k] == train_departure.iloc[i, k]:
            mindopt_gurobi.addConstr(train_stop[i, k] == 0)
        else:
            mindopt_gurobi.addConstr(train_stop[i, k] == 1)
        mindopt_gurobi.addConstr(time_departure[i, k] - time_arrive[i, k] >= train_stop[i, k] * s)

#约束七：行车顺序唯一约束
for i in range(num_station - 1):
    for k in range(num_train):
        for j in range(num_train):
            if k != j:
                mindopt_gurobi.addConstr(train_order[i, k, j] + train_order[i, j, k] == 1)

#约束八：区间最小、最大运行时间约束












#求解
mindopt_gurobi.optimize()
if mindopt_gurobi.status == GRB.OPTIMAL:
    print("最优解已找到，最优目标值为:", mindopt_gurobi.objVal)
    # 提取决策变量信息
    data = []
    for var in mindopt_gurobi.getVars():
        data.append({'Variable Name': var.varName, 'Value': var.x})
    # 创建存储结果的 DataFrame
    data_arrive = pd.DataFrame(index=range(num_station), columns=range(num_train))
    data_departure = pd.DataFrame(index=range(num_station), columns=range(num_train))
    data_stop = pd.DataFrame(index=range(num_station), columns=range(num_train))

    # 填充实际到站时间结果
    for i in range(num_station):
        for k in range(num_train):
            data_arrive.at[i, k] = time_arrive[i, k].x
            data_arrive.iloc[i, k] = minutes_to_time(data_arrive.iloc[i, k])


    # 填充实际发车时间结果
    for i in range(num_station):
        for k in range(num_train):
            data_departure.at[i, k] = time_departure[i, k].x
            data_departure.iloc[i, k] = minutes_to_time(data_departure.iloc[i, k])
    # 填充停站结果
    for i in range(num_station):
        for k in range(num_train):
            data_stop.at[i, k] = train_stop[i, k].x

    """数据整理"""
    data_out = pd.DataFrame(index=range(rows), columns=range(cols))
    # data_out = PLAN
    data_out = PLAN.copy()
    for i in range(num_train):
        for j in range(rows - 1):
            data_out.iloc[j + 1, (i + 1) * 3] = data_arrive.iloc[j, i]
            data_out.iloc[j + 1, (i + 1) * 3 + 1] = data_departure.iloc[j, i]

    """保存文件"""
    file_path = r'D:\mazhen\mindOPT\grurobi\output\output.xlsx'
    data_out.to_excel(file_path, index=False, header=False)
    print("ok")

elif mindopt_gurobi.status == GRB.INFEASIBLE:
    print("模型无解，请检查约束条件。")



