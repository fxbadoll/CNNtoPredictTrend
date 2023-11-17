def single_stock_output_pic():
    # -*- coding: utf-8 -*-
    """
Created on Fri Oct 27 15:56:47 2023
@author: XF
"""

# 导入pandas和numpy库
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
# 确定股票代码

    open_ = pd.read_csv('D:\FuXuan\Coding\单指数\个股行情数据\开盘价.csv',index_col=0)
    close_ = pd.read_csv('D:\FuXuan\Coding\单指数\个股行情数据\收盘价.csv',index_col=0)
    high_ = pd.read_csv('D:\FuXuan\Coding\单指数\个股行情数据\最高价.csv',index_col=0)
    low_ = pd.read_csv('D:\FuXuan\Coding\单指数\个股行情数据\最低价.csv',index_col=0)
    volume_ = pd.read_csv('D:\FuXuan\Coding\单指数\个股行情数据\成交量.csv',index_col=0)

    index_name = volume_.columns
    volume_.columns = open_.columns
    volume_.index = open_.index

# 用5天的数据图来看后5天的涨跌，这个时间参数用time_gap来表示

    time_gap = 20 # 用多长时间的样本数据
    time_gap2 = 5 #看
    moving_avg = 20 #用多长时间的移动平均值
    i = index_name[0]
    single_stock = pd.concat([open_[i],close_[i],high_[i],low_[i],volume_[i]],axis=1)
    single_stock.columns = ['开盘价','收盘价','最高价','最低价','成交量']
    single_stock['日期'] = single_stock.index
    single_stock['日期'] = pd.to_datetime(single_stock['日期'])
    single_stock = single_stock.sort_values('日期')
    single_stock['日期'] = single_stock['日期'].astype(str)
    single_stock['20日均线'] = single_stock['收盘价'].rolling(window=moving_avg).mean()
    single_stock['后5天的涨跌幅'] = single_stock['收盘价'].rolling(window=time_gap).apply(lambda x: (x.iloc[-1]/x.iloc[0])-1)
    single_stock['后5天涨1跌0'] = single_stock['收盘价'].rolling(window=time_gap).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
    single_stock = single_stock.iloc[moving_avg:,:]
    single_stock = single_stock.reset_index(drop=True)
    single_stock = single_stock.sort_values('日期')

    def normalized(data,exclude_column):
    # Define the column to exclude from normalization
    # Create a MinMaxScaler object
        scaler = MinMaxScaler()
    # Fit the scaler to the DataFrame
        scaler.fit(data.drop(exclude_column, axis=1))
    # Transform the DataFrame using the scaler
        normalized_df = pd.DataFrame(scaler.transform(data.drop(exclude_column, axis=1)), 
        columns = data.drop(exclude_column, axis=1).columns)
        normalized_df[exclude_column] = data[exclude_column]
        normalized_df = normalized_df.reset_index(drop=True)
        return normalized_df

# 是否需要将数据先正态化再进行画图
    normal = 0

    if normal:
# 将归一化数据赋值给stock_data
        stock_data = normalized(single_stock, '日期')
    else:
        stock_data = single_stock

# 导入matplotlib库
    import matplotlib.pyplot as plt  
    from PIL import Image

# 将数据保存为csv文件
    single_stock.to_csv("D:\FuXuan\Coding\单指数\股票行情数据\数据整理核对_2.csv")



# 计算后5天涨跌幅的分位数
    len_data = int(stock_data.shape[0])
# 计算后5天涨跌幅的分位数
#quantile = stock_data['后5天的涨跌幅'].rank()/len_data
# 新增一列为后5天涨跌幅的分位数
#stock_data['后5天涨跌幅分位数'] = quantile
    train_label = []
    single_stock = stock_data

    def normalize_pct(data,column):
        df = data.copy()
    # Normalize the first day closing price to one
        df['Norm'] = df[column] / df[column][0]

    # Calculate the percentage change between consecutive values
        df['Returns'] = df[column].pct_change()

    # Set the first row of the 'Returns' column to zero
        df['Returns'][0] = 0

    # Calculate the new closing price by multiplying the normalized price by the percentage change
        df[column] = df['Norm']

        return df[column]
    


# 遍历数据，画图，并保存图片和标签
    for i in range(0,len_data-time_gap-time_gap2+1,time_gap):
    #i = 0
        new_sample = single_stock.iloc[i:i+time_gap,:].copy()
        new_label =  single_stock.iloc[i+time_gap+time_gap2-1,8].copy()
        new_sample = new_sample.reset_index(drop=True)
        for col in new_sample.columns:
            if col not in ['日期','后5天涨1跌0','后5天的涨跌幅'] :
                new_sample[col] = normalize_pct(new_sample,col)

        date_list = new_sample['日期'].tolist()
        start_date = date_list[0]

        b = new_sample.reset_index(drop=True)
        fig = plt.figure(figsize=(20,10))#图象宽和高分别为20和10
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])#创建一个2行1列的网格布局，第一个子图高度是第二个的4倍
    # 第一个子图，K线图
        ax1 = fig.add_subplot(gs[0, 0])
        line_width = 5  # 设置线宽
        ax1.vlines(b['日期'], ymin=b['最低价']-0.01, ymax=b['最高价']+0.01, color='white', linewidth=line_width)#vlines函数在ax1上绘制垂直线段。b['最低价'] 作为线段的起始点的 y 坐标
    #hlines函数在ax1上绘制水平线段
        ax1.hlines(b['开盘价'], xmin=[i-0.2 for i in range(len(b['日期']))], xmax=[i for i in range(len(b['日期']))], color='white', linewidth=line_width)
        ax1.hlines(b['收盘价'], xmin=[i for i in range(len(b['日期']))], xmax=[i+0.2 for i in range(len(b['日期']))], color='white', linewidth=line_width)
        ax1.plot(b['日期'], b['20日均线'], color='white', linewidth=line_width)#画折线图
    # 第二个子图，成交量柱状图
        ax2 = fig.add_subplot(gs[1, 0],sharex=ax1)# 添加 sharex 参数来共享 x 轴
        bar_width = 0.1
        ax2.bar(b['日期'], b['成交量'], color='white', width=bar_width, align='center') 
    # 去掉边框
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
    # 旋转日期标签
        plt.xticks(rotation=45)
    # 去掉横纵坐标
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
    # 调整子图间距
        plt.subplots_adjust(hspace=0)
    # 设置背景色
        fig.patch.set_facecolor('black')
        ax1.set_facecolor('black')
        ax2.set_facecolor('black')

        plt.savefig(f'D:\\FuXuan\\Coding\\单指数\\jd\\images\\image{i:06d}_{start_date}.png',dpi=500)
        plt.close()  
    
        with open(f'D:\\FuXuan\\Coding\\单指数\\jd\\labels\image{i:06d}_{start_date}.txt', 'w') as file:  
    # 将数字写入文件  
            file.write(str(int(new_label)))  
    
        train_label.append(new_label)


if __name__ == '__main__':
    single_stock_output_pic()
