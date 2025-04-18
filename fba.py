import asyncio
import time
import PyQt5.QtWidgets as qt
from PyQt5.QtWidgets import (
    QTableWidgetItem, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QWidget, QLabel, QTableWidget
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QWheelEvent
from ib_insync import IB, util, Stock
import numpy as np
import pandas as pd
import statsmodels.api as sm


###############################################################################
# Helper functions
###############################################################################

def make_duration_str(days: int) -> str:
    """Convert day count to IB-friendly durationStr."""
    if days <= 365:
        return f"{days} D"
    else:
        years = days // 365
        if years < 1:
            years = 1
        return f"{years} Y"


def computeDailyRegressions(df_daily: pd.DataFrame, lb: int):
    """
    1) Slice last `lb` rows of df_daily.
    2) OLS => returns (df_pred, sigma):
       df_pred has columns [lr_value, lr_plus_2, lr_minus_2],
       plus we also return 'sigma' used to define ±2σ.

    If insufficient data => (empty DF, None).
    """
    if df_daily is None or df_daily.empty or len(df_daily) < lb or lb < 5:
        return (pd.DataFrame(), None)

    recent = df_daily.tail(lb).copy()
    closes = recent['close'].values
    X = np.arange(len(closes))
    X = sm.add_constant(X)
    model = sm.OLS(closes, X).fit()
    y_pred = model.predict(X)
    # We estimate sigma from residuals
    residuals = closes - y_pred
    sigma = np.std(residuals)

    df_pred = pd.DataFrame(index=recent.index)
    df_pred['lr_value'] = y_pred
    df_pred['lr_plus_2'] = df_pred['lr_value'] + 2 * sigma
    df_pred['lr_minus_2'] = df_pred['lr_value'] - 2 * sigma
    df_pred.index = pd.to_datetime(df_pred.index)
    return (df_pred, sigma)


def mergeDailyPredictionsInto15Min(df_15m: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Merge df_pred (daily lines) into df_15m (15-min bars) with an as-of merge.
    Both must have a DateTimeIndex. We also ensure they share the same tz.
    """
    if df_15m.empty or df_pred.empty:
        return pd.DataFrame()

    # Match time zones (if df_15m is tz-aware)
    if df_15m.index.tz is not None and df_pred.index.tz is None:
        df_pred.index = pd.to_datetime(df_pred.index).tz_localize(df_15m.index.tz)

    df_15m = df_15m.sort_index()
    df_pred = df_pred.sort_index()

    merged = pd.merge_asof(
        df_15m, df_pred,
        left_index=True,
        right_index=True,
        direction='backward'
    )
    return merged


def findMostRecentHit(merged_15m: pd.DataFrame) -> (str, float):
    """
    Scan from newest to oldest 15-min bar. Check if lr_plus_2, lr_minus_2, or lr_value
    is in [low, high]. Priority: +2σ, -2σ, mean.
    Return (whichLine, lineVal) or (None, None).
    """
    if merged_15m.empty:
        return (None, None)

    for i in range(len(merged_15m) - 1, -1, -1):
        row = merged_15m.iloc[i]
        low_, high_ = row['low'], row['high']
        plus_ = row['lr_plus_2']
        minus_ = row['lr_minus_2']
        mean_ = row['lr_value']

        if low_ <= plus_ <= high_:
            return ("lr_plus_2", plus_)
        if low_ <= minus_ <= high_:
            return ("lr_minus_2", minus_)
        if low_ <= mean_ <= high_:
            return ("lr_value", mean_)

    return (None, None)


def compareToCurrentBar(merged_15m: pd.DataFrame, whichLine: str, lineVal: float) -> str:
    """
    Compare lineVal to the *latest* 15-min bar's average to get an indicator => a string like "++119.97" or "--122.20", etc.
    If no line is found => "N/A".
    """
    if not whichLine or merged_15m.empty:
        return "N/A"

    latest = merged_15m.iloc[-1]
    avg_15m = (latest['open'] + latest['high'] + latest['low'] + latest['close']) / 4.0

    lineVal = np.round(lineVal, 2)
    # We'll show e.g. "++120.33" if current bar < lineVal, or "--120.33" if bar > lineVal
    if whichLine in ('lr_plus_2', 'lr_minus_2'):
        return f"-- {lineVal}" if (avg_15m > lineVal) else f"++ {lineVal}"
    else:
        # lr_value
        return f"- {lineVal}" if (avg_15m > lineVal) else f"+ {lineVal}"




###############################################################################
# TickerTable
###############################################################################
class TickerTable(qt.QTableWidget):

    headers = [
        'Symbol', 'Last',
        'ShortLB', 'MedLB', 'LongLB',
        'ShortSignal', 'MediumSignal', 'LongSignal'
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.conId2Row = {}

        self._baseFontSize = 10
        font = self.font()
        font.setPointSize(self._baseFontSize)
        self.setFont(font)

        self.setColumnCount(len(self.headers))
        self.setHorizontalHeaderLabels(self.headers)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(qt.QAbstractItemView.DoubleClicked)

    def wheelEvent(self, event: QWheelEvent):
        """
        If Ctrl is held, interpret the wheel as zoom in/out.
        Otherwise, do normal scrolling.
        """
        modifiers = qt.QApplication.keyboardModifiers()
        if modifiers & Qt.ControlModifier:
            angleDeltaY = event.angleDelta().y()
            if angleDeltaY > 0:
                # zoom in
                self._baseFontSize += 1
            else:
                # zoom out
                if self._baseFontSize > 5:
                    self._baseFontSize -= 1

            # apply new font
            font = self.font()
            font.setPointSize(self._baseFontSize)
            self.setFont(font)

            # optionally auto-resize
            self.resizeColumnsToContents()
            self.resizeRowsToContents()

            event.accept()
        else:
            # normal scroll
            super().wheelEvent(event)

    def __contains__(self, contract):
        return contract.conId in self.conId2Row

    def addTickerRow(self, ticker, shortLB=20, medLB=150, longLB=220):
        conId = ticker.contract.conId
        row = self.rowCount()
        self.insertRow(row)
        self.conId2Row[conId] = row

        for col in range(len(self.headers)):
            item = QTableWidgetItem('-')
            if col in (2, 3, 4):  # shortLB, medLB, longLB
                item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.setItem(row, col, item)

        self.item(row, 0).setText(ticker.contract.symbol)  # Symbol
        self.item(row, 1).setText('-')  # Last
        self.item(row, 2).setText(str(shortLB))
        self.item(row, 3).setText(str(medLB))
        self.item(row, 4).setText(str(longLB))

        self.resizeColumnsToContents()

    def onPendingTickers(self, tickers, lr_dict):
        """
        Update the Last column and the ±σ suffixes for the Short/Medium/Long
        signal columns.  We use the best available price:
            real‑time last   →   yesterday's close   →   prevClose snapshot
        """
        for t in tickers:
            row = self.conId2Row.get(t.contract.conId)
            if row is None:
                continue

            # 1.  Decide which price to use -------------------------------
            price = t.last
            if price is None or np.isnan(price):
                price = t.close if t.close is not None and not np.isnan(t.close) else None
            if price is None or np.isnan(price):
                price = t.prevClose if t.prevClose is not None and not np.isnan(t.prevClose) else None

            # 2.  Update the “Last” column if we have a price -------------
            if price is not None and not np.isnan(price):
                self.item(row, 1).setText(f"{price:.2f}")

            # 3.  Always append the ±σ differences when possible ----------
            if price is not None and not np.isnan(price):
                lr_info = lr_dict.get(t.contract.conId)
                if lr_info:
                    shortVal, shortSig = lr_info.get("short", (None, None))
                    medVal, medSig = lr_info.get("medium", (None, None))
                    longVal, longSig = lr_info.get("long", (None, None))

                    # columns: 5 = ShortSignal, 6 = MediumSignal, 7 = LongSignal
                    self._appendSigmaDiff(row, 5, price, shortVal, shortSig)
                    self._appendSigmaDiff(row, 6, price, medVal, medSig)
                    self._appendSigmaDiff(row, 7, price, longVal, longSig)

        # keep the table neat
        self.resizeColumnsToContents()

    def _appendSigmaDiff(self, row: int, col: int, lastPrice: float, lrVal: float, sigma: float):
        """
        1. Take the existing cell text, e.g. "++119.97" or "N/A" or "...".
        2. Remove any old " | ±Xσ" suffix if it exists (split on '|').
        3. If we have valid lrVal/sigma => compute diffInSigma = (lastPrice - lrVal)/sigma => e.g. +2.0σ
        4. Rebuild the final string as "BaseSignal | +2.0σ"
        """
        item = self.item(row, col)
        if not item:
            return

        base_text = item.text()
        # if there's a '|', keep only what's before it
        if '|' in base_text:
            base_text = base_text.split('|')[0].strip()

        # If we can't compute a valid diff, just keep the base
        if (lrVal is None) or (sigma is None) or (sigma == 0.0):
            new_text = base_text
        else:
            diff_sigma = (lastPrice - lrVal) / sigma
            diff_str = f"{diff_sigma:+.2f}σ"
            new_text = f"{base_text} | {diff_str}"

        item.setText(new_text)

    def setSignals(self, conId, shortSig, medSig, longSig):
        """
        After "Compute Signals", we set the textual signals for short/med/long
        (e.g. "++119.97"). We'll later append the real-time "±Xσ" in onPendingTickers.
        """
        row = self.conId2Row.get(conId)
        if row is None:
            return
        # columns 5,6,7 are short/medium/long signal
        self.item(row, 5).setText(shortSig)
        self.item(row, 6).setText(medSig)
        self.item(row, 7).setText(longSig)

    def clearTickers(self):
        self.setRowCount(0)
        self.conId2Row.clear()


###############################################################################
# MainWindow
###############################################################################
class MainWindow(qt.QWidget):
    """
    1) "Update Regressions" => fetch daily bars, store short/med/long results
       in self.regressions_dict plus final (lr_value, sigma) in self.lrLatest. In that function a part of the indicator
    2) "Compute Signals" => fetch 15-min bars => short/med/long signals in table.
    3) "onPendingTickers" => Besides updating LastPrice column live, also show ±Xσ difference appended to short/med/long signals.
    """

    def __init__(self, host='127.0.0.1', port=7497, clientId=1):
        super().__init__()
        self.ib = IB()
        self.ib.pendingTickersEvent += self.onPendingTickers

        self.connectInfo = (host, port, clientId)

        # Regressions: self.regressions_dict[conId] = {"shortPred":..., "medPred":..., "longPred":...}
        self.regressions_dict = {}
        # Latest LR (lr_value + sigma) for short/med/long => used for ±Xσ difference
        self.lrLatest = {}


        # UI
        self.fontSize = 10
        self.zoomInBtn = QPushButton("Zoom In")
        self.zoomOutBtn = QPushButton("Zoom Out")
        self.zoomInBtn.clicked.connect(self.onZoomIn)
        self.zoomOutBtn.clicked.connect(self.onZoomOut)

        self.connectBtn = QPushButton("Connect")
        self.connectBtn.clicked.connect(self.onConnectClicked)

        self.addLabel = QLabel("Add US Equity:")
        self.addEdit = QLineEdit("")
        self.addButton = QPushButton("Add Ticker")
        self.addButton.clicked.connect(self.onAddTicker)

        self.table = TickerTable()

        self.updateRegBtn = QPushButton("Update Regressions")
        self.updateRegBtn.clicked.connect(self.onUpdateRegressions)

        self.signalsBtn = QPushButton("Compute Signals")
        self.signalsBtn.clicked.connect(self.onComputeSignals)

        self.addPortfolioButton = QPushButton("Add Portfolio tickers")
        self.addPortfolioButton.clicked.connect(self.onAddPortfolioTickers)

        # Layout
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.connectBtn)
        topLayout.addWidget(self.addLabel)
        topLayout.addWidget(self.addEdit)
        topLayout.addWidget(self.addButton)
        topLayout.addWidget(self.addPortfolioButton)

        mainLayout = QVBoxLayout(self)
        mainLayout.addLayout(topLayout)
        mainLayout.addWidget(self.table)

        bottomLayout = QHBoxLayout()
        bottomLayout.addWidget(self.updateRegBtn)
        bottomLayout.addWidget(self.signalsBtn)
        mainLayout.addLayout(bottomLayout)

        self.setWindowTitle("price_movement signal from Above/Below LinReg_bands | with Real-Time ±σ values from Regression")

        # QTimer for auto signals
        self.timer = QTimer()
        self.timer.setInterval(15 * 60 * 1000)  # 15 min
        self.timer.timeout.connect(self.onComputeSignals)
        # self.timer.start()  # optionally start

    def onZoomIn(self):
        self.fontSize += 1
        self.applyZoom()

    def onZoomOut(self):
        if self.fontSize > 5:
            self.fontSize -= 1
        self.applyZoom()

    def applyZoom(self):
        """
        Update the QTableWidget's font (and optionally row/column sizes)
        to match self.fontSize.
        """
        font = self.table.font()
        font.setPointSize(self.fontSize)
        self.table.setFont(font)

        # resize rows/columns automatically or fix them
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

    def onConnectClicked(self):
        if self.ib.isConnected():
            self.ib.disconnect()
            self.table.clearTickers()
            self.connectBtn.setText("Connect")
        else:
            self.ib.connect(*self.connectInfo)
            self.ib.reqMarketDataType(1)  # live
            self.connectBtn.setText("Disconnect")

    def onAddTicker(self):
        sym = self.addEdit.text().strip().upper()
        if not sym:
            return
        if not self.ib.isConnected():
            print("Not connected.")
            return
        contract = Stock(sym, 'SMART', 'USD')
        c = self.ib.qualifyContracts(contract)
        if c:
            ticker = self.ib.reqMktData(c[0], '', False, False)
            self.table.addTickerRow(ticker, shortLB=20, medLB=120, longLB=220)
        self.addEdit.clear()

    def onPendingTickers(self, tickers):
        """
        Called on every real-time price update. We'll pass self.lrLatest so that
        the table can append "±Xσ" differences to the existing Short/Med/Long signals.
        """
        self.table.onPendingTickers(tickers, self.lrLatest)

    def onAddPortfolioTickers(self):
        if not self.ib.isConnected():
            print("Not connected.")
            return
        # Get portfolio items; ib.portfolio() returns a list of PortfolioItem objects.
        portfolioItems = self.ib.portfolio()
        for item in portfolioItems:
            contract = item.contract
            # Skip if already added
            if contract.conId in self.table.conId2Row:
                continue
            # Qualify the contract and request market data
            c = self.ib.qualifyContracts(contract)
            if c:
                ticker = self.ib.reqMktData(c[0], '', False, False)
                # You can adjust the LB values as needed
                self.table.addTickerRow(ticker, shortLB=20, medLB=120, longLB=220)


    ###########################################################################
    # Step 1: "Update Regressions"
    ###########################################################################
    def onUpdateRegressions(self):
        if not self.ib.isConnected():
            print("Not connected.")
            return

        for conId, rowIdx in self.table.conId2Row.items():
            # find Ticker
            found = None
            for t in self.ib.tickers():
                if t.contract.conId == conId:
                    found = t
                    break
            if not found:
                continue
            contract = found.contract

            # read shortLB, medLB, longLB
            shortLB_str = self.table.item(rowIdx, 2).text()
            medLB_str   = self.table.item(rowIdx, 3).text()
            longLB_str  = self.table.item(rowIdx, 4).text()
            try:
                shortLB = int(shortLB_str)
                medLB   = int(medLB_str)
                longLB  = int(longLB_str)
            except ValueError:
                self.table.setSignals(conId, "Err", "Err", "Err")
                continue

            maxLB = max(shortLB, medLB, longLB)
            if maxLB < 5:
                self.table.setSignals(conId, "NoData", "NoData", "NoData")
                continue

            days_needed = maxLB * 2
            durationStr = make_duration_str(days_needed)
            print(f"Updating daily regs for {contract.symbol} => short={shortLB}, med={medLB}, long={longLB}, dur={durationStr}")

            try:
                daily_bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr=durationStr,
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=False,
                    keepUpToDate=False
                )
            except asyncio.TimeoutError:
                self.table.setSignals(conId, "Timeout", "Timeout", "Timeout")
                continue

            df_daily = util.df(daily_bars)
            if df_daily.empty or len(df_daily) < 5:
                self.table.setSignals(conId, "NoData", "NoData", "NoData")
                continue

            df_daily.set_index('date', inplace=True)

            df_short, sigma_short = computeDailyRegressions(df_daily, shortLB)
            df_med,   sigma_med   = computeDailyRegressions(df_daily, medLB)
            df_long,  sigma_long  = computeDailyRegressions(df_daily, longLB)

            if df_short.empty or df_med.empty or df_long.empty:
                self.table.setSignals(conId, "NoData", "NoData", "NoData")
                continue

            # Store the data in memory
            self.regressions_dict[conId] = {
                "shortPred": df_short,
                "medPred":   df_med,
                "longPred":  df_long
            }

            # For real-time ±σ diffs:
            # We'll take the last row's lr_value from each DF, and the sigma from each.
            # If the DF has at least 1 row, let's do:
            shortVal = df_short.iloc[-1]['lr_value']
            medVal   = df_med.iloc[-1]['lr_value']
            longVal  = df_long.iloc[-1]['lr_value']

            self.lrLatest[conId] = {
                "short":  (shortVal, sigma_short),
                "medium": (medVal,   sigma_med),
                "long":   (longVal,  sigma_long)
            }

            print(f"Updated regressions for {contract.symbol}")

    ###########################################################################
    # Step 2: "Compute Signals"
    ###########################################################################
    def onComputeSignals(self):
        """
        For each ticker, fetch 15-min bars => merges => find hits => set short/med/long signals.
        The real-time difference in multiples of σ is appended in onPendingTickers (above).
        """
        if not self.ib.isConnected():
            print("Not connected.")
            return

        for conId, rowIdx in self.table.conId2Row.items():
            # find Ticker
            found = None
            for t in self.ib.tickers():
                if t.contract.conId == conId:
                    found = t
                    break
            if not found:
                continue
            contract = found.contract

            preds = self.regressions_dict.get(conId)
            if not preds:
                self.table.setSignals(conId, "UpdFirst", "UpdFirst", "UpdFirst")
                continue

            df_short = preds["shortPred"]
            df_med   = preds["medPred"]
            df_long  = preds["longPred"]
            if df_short.empty or df_med.empty or df_long.empty:
                self.table.setSignals(conId, "NoData", "NoData", "NoData")
                continue

            print(f"Compute Signals => {contract.symbol}")
            try:
                bars_15 = self.ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='25 D',
                    barSizeSetting='15 mins',
                    whatToShow='TRADES',
                    useRTH=False,
                    keepUpToDate=False
                )
            except asyncio.TimeoutError:
                self.table.setSignals(conId, "Timeout", "Timeout", "Timeout")
                continue

            df15 = util.df(bars_15)
            if df15.empty:
                self.table.setSignals(conId, "No15m", "No15m", "No15m")
                continue

            df15.set_index('date', inplace=True)

            # short
            merged_s = mergeDailyPredictionsInto15Min(df15, df_short)
            b_s, v_s = findMostRecentHit(merged_s)
            shortSig = compareToCurrentBar(merged_s, b_s, v_s)

            # medium
            merged_m = mergeDailyPredictionsInto15Min(df15, df_med)
            b_m, v_m = findMostRecentHit(merged_m)
            medSig = compareToCurrentBar(merged_m, b_m, v_m)

            # long
            merged_l = mergeDailyPredictionsInto15Min(df15, df_long)
            b_l, v_l = findMostRecentHit(merged_l)
            longSig = compareToCurrentBar(merged_l, b_l, v_l)

            self.table.setSignals(conId, shortSig, medSig, longSig)

    def closeEvent(self, event):
        loop = util.getLoop()
        loop.stop()


###############################################################################
# Launch
###############################################################################
if __name__ == '__main__':
    util.patchAsyncio()
    util.useQt()

    window = MainWindow('127.0.0.1', 7497, 1)
    window.resize(900, 500)
    window.show()

    # Optionally start auto signals:
    window.timer.start()

    IB.run()
