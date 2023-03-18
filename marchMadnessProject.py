import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as web
import yfinance as yf
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import sklearn.metrics
import seaborn as sb
from sklearn.metrics import r2_score

march = pd.read_csv('tournament_game_data.csv',index_col='ID')
march.drop(['TEAM','TEAM2', 'SCORE', 'FREE THROW % DEFENSE',
            'EFG % DEFENSE', 'FREE THROW RATE DEFENSE',
            '3PT RATE DEFENSE', 'OP ASSIST %', 'OP O REB %',
            'OP D REB %', 'BLOCKED %', 'TURNOVER % DEFENSE', '2PT %', 'EFG %',
            'WINS ABOVE BUBBLE', 'BARTHAG', 'OFFENSIVE REBOUND %', 'ELITE SOS',
            'DEFENSIVE REBOUND %', 'BLOCK %', '2PT % DEFENSE', '3PT % DEFENSE'
            ],axis=1,inplace=True)

values = [32,16,8,4,2,0]
march = march[march.CurrentRound.isin(values) == False]

pd.set_option('display.max_columns',5)
# print(march)


def printOutTheCoefficients(params,coeffecients,intercept):
    tParams = params[np.newaxis].T
    tCoeffs = coeffecients.T
    total = np.concatenate([tParams,tCoeffs],axis=1)
    totalDF = pd.DataFrame(data=total)
    totalDF.to_excel("modelOutput.xlsx")
    print(totalDF)

def WonChampionship(x):
    if x == 1:
        return 'Champion'
    else:
        return 'Loser'
march['WonChampionship'] = march.apply(lambda row: WonChampionship(row['TEAM ROUND']),axis=1)

def ChampionshipGame(x):
    if x == 64:
        return 'No Championship Game'
    if x == 32:
        return 'No Championship Game'
    if x == 16:
        return 'No Championship Game'
    if x == 8:
        return 'No Championship Game'
    if x == 4:
        return 'No Championship Game'
    else:
        return 'Championship Game'
march['ChampionshipGame'] = march.apply(lambda row: ChampionshipGame(row['TEAM ROUND']),axis=1)

def FinalFour(x):
    if x == 64:
        return 'No Final Four'
    if x == 32:
        return 'No Final Four'
    if x == 16:
        return 'No Final Four'
    if x == 8:
        return 'No Final Four'
    else:
        return 'Final Four'
march['FinalFour'] = march.apply(lambda row: FinalFour(row['TEAM ROUND']),axis=1)

def Elite8(x):
    if x == 64:
        return 'No Elite 8'
    if x == 32:
        return 'No Elite 8'
    if x == 16:
        return 'No Elite 8'
    else:
        return 'Elite 8'
march['Elite8'] = march.apply(lambda row: Elite8(row['TEAM ROUND']),axis=1)

def Sweet16(x):
    if x == 64:
        return 'No Sweet 16'
    if x == 32:
        return 'No Sweet 16'
    else:
        return 'Sweet 16'
march['Sweet16'] = march.apply(lambda row: Sweet16(row['TEAM ROUND']),axis=1)

def RoundOf32(x):
    if x == 64:
        return 'No Round of 32'
    else:
        return 'Round of 32'
march['RoundOf32'] = march.apply(lambda row: RoundOf32(row['TEAM ROUND']),axis=1)
march.drop(['TEAM ROUND'],axis=1,inplace=True)

kpeff = march['KENPOM ADJUSTED EFFICIENCY'].mean()
# print(kpeff)

def KPEff(x):
    if x > kpeff:
        return 'Above Average Ken Pom Eff'
    else:
        return 'Below Average Ken Pom Eff'
march['KPEff'] = march.apply(lambda row: KPEff(row['KENPOM ADJUSTED EFFICIENCY']),axis=1)
march.drop(['KENPOM ADJUSTED EFFICIENCY'],axis=1,inplace=True)

kpoff = march['KENPOM ADJUSTED OFFENSE'].mean()

def KPOff(x):
    if x > kpoff:
        return 'Above Average Ken Pom Off'
    else:
        return 'Below Average Ken Pom Off'
march['KPOff'] = march.apply(lambda row: KPOff(row['KENPOM ADJUSTED OFFENSE']),axis=1)
march.drop(['KENPOM ADJUSTED OFFENSE'],axis=1,inplace=True)

kpdef = march['KENPOM ADJUSTED DEFENSE'].mean()

def KPDef(x):
    if x > kpdef:
        return 'Above Average Ken Pom Def'
    else:
        return 'Below Average Ken Pom Def'
march['KPDef'] = march.apply(lambda row: KPDef(row['KENPOM ADJUSTED DEFENSE']),axis=1)
march.drop(['KENPOM ADJUSTED DEFENSE'],axis=1,inplace=True)

kptempo = march['KENPOM ADJUSTED TEMPO'].mean()

def KPTempo(x):
    if x > kptempo:
        return 'Above Average Ken Pom Tempo'
    else:
        return 'Below Average Ken Pom Tempo'
march['KPTempo'] = march.apply(lambda row: KPTempo(row['KENPOM ADJUSTED TEMPO']),axis=1)
march.drop(['KENPOM ADJUSTED TEMPO'],axis=1,inplace=True)

bteff = march['BARTTORVIK ADJUSTED EFFICIENCY'].mean()

def BTEff(x):
    if x > bteff:
        return 'Above Average Bart Torvik Eff'
    else:
        return 'Below Average Bart Torvik Eff'
march['BTEff'] = march.apply(lambda row: BTEff(row['BARTTORVIK ADJUSTED EFFICIENCY']),axis=1)
march.drop(['BARTTORVIK ADJUSTED EFFICIENCY'],axis=1,inplace=True)

btoff = march['BARTTORVIK ADJUSTED OFFENSE'].mean()

def BTOff(x):
    if x > btoff:
        return 'Above Average Bart Torvik Off'
    else:
        return 'Below Average Bart Torvik Off'
march['BTOff'] = march.apply(lambda row: BTOff(row['BARTTORVIK ADJUSTED OFFENSE']),axis=1)
march.drop(['BARTTORVIK ADJUSTED OFFENSE'],axis=1,inplace=True)

btdef = march['BARTTORVIK ADJUSTED DEFENSE'].mean()

def BTDef(x):
    if x > btdef:
        return 'Above Average Bart Torvik Def'
    else:
        return 'Below Average Bart Torvik Def'
march['BTDef'] = march.apply(lambda row: BTDef(row['BARTTORVIK ADJUSTED DEFENSE']),axis=1)
march.drop(['BARTTORVIK ADJUSTED DEFENSE'],axis=1,inplace=True)

bttempo = march['BARTTORVIK ADJUSTED TEMPO'].mean()

def BTTempo(x):
    if x > bttempo:
        return 'Above Average Bart Torvik Tempo'
    else:
        return 'Below Average Bart Torvik Tempo'
march['BTTempo'] = march.apply(lambda row: BTTempo(row['BARTTORVIK ADJUSTED TEMPO']),axis=1)
march.drop(['BARTTORVIK ADJUSTED TEMPO'],axis=1,inplace=True)

threepct = march['3PT %'].mean()

def ThreePCT(x):
    if x > threepct:
        return 'Above Average 3 PT %'
    else:
        return 'Below Average 3 PT %'
march['ThreePCT'] = march.apply(lambda row: ThreePCT(row['3PT %']),axis=1)
march.drop(['3PT %'],axis=1,inplace=True)

ftpct = march['FREE THROW %'].mean()

def FtPCT(x):
    if x > ftpct:
        return 'Above Average FT %'
    else:
        return 'Below Average FT %'
march['FtPCT'] = march.apply(lambda row: FtPCT(row['FREE THROW %']),axis=1)
march.drop(['FREE THROW %'],axis=1,inplace=True)

threerate = march['3PT RATE'].mean()

def ThreeRate(x):
    if x > threerate:
        return 'Above Average 3 PT Rate'
    else:
        return 'Below Average 3 PT Rate'
march['ThreeRate'] = march.apply(lambda row: ThreeRate(row['3PT RATE']),axis=1)
march.drop(['3PT RATE'],axis=1,inplace=True)

ftrate = march['FREE THROW RATE'].mean()

def FtRate(x):
    if x > ftrate:
        return 'Above Average FT Rate'
    else:
        return 'Below Average FT Rate'
march['FtRate'] = march.apply(lambda row: FtRate(row['FREE THROW RATE']),axis=1)
march.drop(['FREE THROW RATE'],axis=1,inplace=True)

assistpct = march['ASSIST %'].mean()

def AssistPCT(x):
    if x > assistpct:
        return 'Above Average Assist %'
    else:
        return 'Below Average Assist %'
march['AssistPCT'] = march.apply(lambda row: AssistPCT(row['ASSIST %']),axis=1)
march.drop(['ASSIST %'],axis=1,inplace=True)

turnoverpct = march['TURNOVER %'].mean()

def TurnoverPCT(x):
    if x > turnoverpct:
        return 'Above Average Turnover %'
    else:
        return 'Below Average Turnover %'
march['TurnoverPCT'] = march.apply(lambda row: TurnoverPCT(row['TURNOVER %']),axis=1)
march.drop(['TURNOVER %'],axis=1,inplace=True)

pppoff = march['POINTS PER POSSESSION OFFENSE'].mean()

def PPPOff(x):
    if x > pppoff:
        return 'Above Average Offensive Points/Possession'
    else:
        return 'Below Average Offensive Points/Possession'
march['PPPOff'] = march.apply(lambda row: PPPOff(row['POINTS PER POSSESSION OFFENSE']),axis=1)
march.drop(['POINTS PER POSSESSION OFFENSE'],axis=1,inplace=True)

pppdef = march['POINTS PER POSSESSION DEFENSE'].mean()

def PPPDef(x):
    if x > pppdef:
        return 'Above Average Defensive Points/Possession'
    else:
        return 'Below Average Defensive Points/Possession'
march['PPPDef'] = march.apply(lambda row: PPPDef(row['POINTS PER POSSESSION DEFENSE']),axis=1)
march.drop(['POINTS PER POSSESSION DEFENSE'],axis=1,inplace=True)

winpct = march['WIN %'].mean()

def WinPCT(x):
    if x > winpct:
        return 'Above Average Win %'
    else:
        return 'Below Average Win %'
march['WinPCT'] = march.apply(lambda row: WinPCT(row['WIN %']),axis=1)
march.drop(['WIN %'],axis=1,inplace=True)

march.drop(['CurrentRound','YEAR'],axis=1,inplace=True)
# march.to_csv('cleanMarchMadnessData.csv')

champion = pd.get_dummies(march['WonChampionship'])
finalfour = pd.get_dummies(march['FinalFour'])
elite8 = pd.get_dummies(march['Elite8'])
sweet16 = pd.get_dummies(march['Sweet16'])
roundof32 = pd.get_dummies(march['RoundOf32'])
kpefficiency = pd.get_dummies(march['KPEff'])
kpoffense = pd.get_dummies(march['KPOff'])
kpdefense = pd.get_dummies(march['KPDef'])
kptemp = pd.get_dummies(march['KPTempo'])
btefficiency = pd.get_dummies(march['BTEff'])
btoffense = pd.get_dummies(march['BTOff'])
btdefense = pd.get_dummies(march['BTDef'])
bttemp = pd.get_dummies(march['BTTempo'])
threeperc = pd.get_dummies(march['ThreePCT'])
ftperc = pd.get_dummies(march['FtPCT'])
threert = pd.get_dummies(march['ThreeRate'])
ftrt = pd.get_dummies(march['FtRate'])
assistperc = pd.get_dummies(march['AssistPCT'])
turnoverperc = pd.get_dummies(march['TurnoverPCT'])
pppoffense = pd.get_dummies(march['PPPOff'])
pppdefense = pd.get_dummies(march['PPPDef'])
winperc = pd.get_dummies(march['WinPCT'])
seed = pd.get_dummies(march['SEED'])


march.drop('SEED',axis=1,inplace=True)
march.drop('WonChampionship',axis=1,inplace=True)
march.drop('ChampionshipGame',axis=1,inplace=True)
march.drop('FinalFour',axis=1,inplace=True)
march.drop('Elite8',axis=1,inplace=True)
march.drop('Sweet16',axis=1,inplace=True)
march.drop('RoundOf32',axis=1,inplace=True)
march.drop('KPEff',axis=1,inplace=True)
march.drop('KPOff',axis=1,inplace=True)
march.drop('KPDef',axis=1,inplace=True)
march.drop('KPTempo',axis=1,inplace=True)
march.drop('BTEff',axis=1,inplace=True)
march.drop('BTOff',axis=1,inplace=True)
march.drop('BTDef',axis=1,inplace=True)
march.drop('BTTempo',axis=1,inplace=True)
march.drop('ThreePCT',axis=1,inplace=True)
march.drop('FtPCT',axis=1,inplace=True)
march.drop('ThreeRate',axis=1,inplace=True)
march.drop('FtRate',axis=1,inplace=True)
march.drop('AssistPCT',axis=1,inplace=True)
march.drop('TurnoverPCT',axis=1,inplace=True)
march.drop('PPPOff',axis=1,inplace=True)
march.drop('PPPDef',axis=1,inplace=True)
march.drop('WinPCT',axis=1,inplace=True)

marchJoin = pd.concat([champion,kpefficiency,kpdefense,
                       kpoffense, kptemp, btefficiency, btdefense, btoffense, bttemp,
                       threeperc, ftperc, threert, ftrt, assistperc, turnoverperc,
                       pppoffense, pppdefense, winperc, seed],axis=1)
# print(marchJoin)
marchJoin.to_excel('marchJoinedDF.xlsx')

correlation = marchJoin.corr()
sb.heatmap(correlation)
plt.show()
correlation.to_excel('corr.xlsx')

dfResults = marchJoin['Champion']
dfInputs = marchJoin.drop(['Champion','Loser'],axis=1)
dfInputs.to_excel('inputs.xlsx')

inputsTrain,inputsTest,resultTrain,resultTest = train_test_split(dfInputs,dfResults,test_size=0.3,random_state=1)

LogReg = LogisticRegression()
LogReg.fit(inputsTrain,resultTrain)

resultPred = LogReg.predict(inputsTest)
print(classification_report(resultTest,resultPred))
print("Intercept(b):", LogReg.intercept_)

printOutTheCoefficients(dfInputs.columns.values,LogReg.coef_,LogReg.intercept_)