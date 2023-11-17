
# 本程序是用于回测策略，给定日期



import pandas as pd
from datetime import datetime
import backtrader as bt
import backtrader.feeds as btfeeds
import tushare as ts
import backtrader_plotting

# 利用tushare来提取数据，先用一支个股的数据
# 设置tushare的token
ts.set_token('6e2bea2e792967b0b7c39361d2257c952cf5bb8a019935b905f9aedb')

# 初始化pro接口
pro = ts.pro_api()

# 提取个股数据
df = pro.daily(ts_code='000300.SH', start_date='20201109', end_date='20231107')
df.to_csv('D:\\FuXuan\\Coding\\单指数\回测数据\\stock_data.csv',index=False)


# 先读入回测的个股数据，包括开盘，收盘，最高价，最低价，交易量，处理日期，将日期转换成同一个格式
df = pd.read_csv('D:\\FuXuan\\Coding\\单指数\回测数据\\stock_data.csv')
df.trade_date = df.trade_date.astype("str").astype("datetime64")
df.set_index('trade_date',inplace=True,drop=False)
df = df.sort_index(ascending=True)

data = btfeeds.PandasData(
    dataname=df,
    fromdate=datetime(2020, 11, 9),
    todate=datetime(2023, 11, 7),
    datetime='trade_date', 
    close='close',
    open='open',
    high='high',
    low='low',
    volume='vol',
    openinterest=-1
)

#读入信号数据的时候，要注意日期格式保持YYYY-MM-DD的格式

signal = pd.read_csv('D:\\FuXuan\\Coding\\单指数\回测数据\\signal.csv')
signal_buy = signal[signal.signal == 1]
buy_date = signal_buy.trade_date.tolist()


#signal_sell = signal[signal.signal == 0]
#sell_date = signal_sell.trade_date.tolist()


class TestStrategy(bt.Strategy):
    params = (
        ('percent', 0.99),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        print("Initialization")
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
     
            # Not yet ... we MIGHT BUY if ...
            if str(self.datetime.date(0)) in buy_date:


                # BUY, BUY, BUY!!! (with default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.order_target_percent(target = self.params.percent)

            #if str(self.datetime.date(0)) in sell_date:

                # 如果收到做空信号，那就进行卖空操作，然后在5天后平仓
             #   self.log('SELL CREATE, %.2f' % self.dataclose[0])
              #  self.order = self.order_target_percent(target = -self.params.percent)

        else:

            # Already in the market ... we might sell
            if len(self) >= (self.bar_executed + 5):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('平仓 CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.order_target_percent(target = 0)


if __name__ == '__main__':

    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(1000000)
    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='TradeAnalyzer')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='AnnualReturn')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='Return')
    # Run over everything
    results = cerebro.run()

    sharpe_ratio = results[0].analyzers.sharpe.get_analysis()
    print('Sharpe Ratio:', sharpe_ratio['sharperatio'])

    drawdown = results[0].analyzers.drawdown.get_analysis()
    print('Max Drawdown:', drawdown['max']['drawdown'])

    TradeAnalyzer = results[0].analyzers.TradeAnalyzer.get_analysis()
    print('Total number of trades:', TradeAnalyzer['total']['total'])
    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.plot(iplot=False)

    
