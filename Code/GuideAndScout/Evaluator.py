
from Runner.EvalRunner import EvalRunner
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from const import evalNoiseLevels
import os


class Evaluator():
    def __init__(self, envType) -> None:
        self.envType = envType
        self.modelToEvaluate = []
        for name in os.listdir(f"./Saves/HyperParam/{envType}"):
            # print(name)
            saveName = f"HyperParam/{envType}/{name}"
            model = (saveName, 0, name)
            self.modelToEvaluate.append(model)
        # self.modelToEvaluate = self.modelToEvaluate[1:10]
        self.models = self.modelToEvaluate
        self.repetitions = 100
        self.savePath = f"./Saves/Evaluation/{self.envType}/"

    def testRun(self, run: EvalRunner, noiseP, noiseHandlingMode):
        steps, reward = run.test(
            noiseP=noiseP, noiseHandlingMode=noiseHandlingMode)
        return steps, reward

    def modelEval(self, model, reEval):
        """Evaluate specified model and storing experiment results"""
        modelName = model[0]
        noiseHandling = model[1]
        saveName = model[2]
        if reEval or saveName not in os.listdir(self.savePath):
            fullSavePath = self.savePath + saveName
            os.mkdir(fullSavePath)
            run = EvalRunner(self.envType, modelName)
            print(f"Evaluating Model {saveName}:")
            epsDf = []
            rwdDf = []
            for noise in tqdm(evalNoiseLevels):
                testNoise = noise
                if saveName == "Norm" or noise == 0:
                    testNoise = 0
                for _ in range(self.repetitions):
                    epsE, epsR = self.testRun(run, testNoise, noiseHandling)
                    epsDf.append([noise, epsE])
                    rwdDf.append([noise, epsR])

            epsDf = pd.DataFrame(epsDf, columns=['Noise', 'Eps'])
            rwdDf = pd.DataFrame(rwdDf, columns=['Noise', 'Rwd'])

            dump(
                epsDf, f"{fullSavePath}/Eps")
            dump(
                rwdDf, f"{fullSavePath}/Rwd")

    def findBest(self, best):
        """Search for the best performing models"""
        models = {}
        for model in os.listdir(self.savePath):
            if model != "Norm_Noised" and model != "Norm":
                epsRecord = load(f"{self.savePath}{model}/Eps")
                groupMean = epsRecord["Eps"].mean()
                models[model] = groupMean
        topModel = [k for k in sorted(models, key=models.get)][:best]
        return (topModel)

    def plotBest(self, best):
        models = self.findBest(best)
        self.doPlot("Eps", models)

    def plotAll(self):
        """Plot results of all models in save path"""
        models = os.listdir(self.savePath)
        if "Norm" in models:
            models.remove("Norm")
        if "Norm_Noised" in models:
            models.remove("Norm_Noised")
        self.doPlot("Eps", models)

    def doPlot(self, plotType, models):
        for modelSaveName in models:
            epsRecord = load(
                f"{self.savePath}{modelSaveName}/{plotType}")
            style = None
            if modelSaveName == "Norm":
                style = "dashed"
            sns.lineplot(data=epsRecord, x="Noise",
                         y=plotType, label=modelSaveName, linestyle=style)
            # plt.plot(self.noiseLevels, epsRecord,
            #          label=modelSaveName, linestyle=style)
        plt.xlabel("Noise Level (p)")
        plt.ylabel("Average steps per episode")
        plt.legend(loc="upper left")
        plt.show()

    def normNoiseCompare(self):
        """Generates comparison figure between the top model, checksum, norm and norm_noised"""
        bestModel = self.findBest(1)[0]
        print(bestModel)
        models = f"HyperParam/{self.envType}/{bestModel}"
        normNoisedModel = (models, None, "Norm_Noised")
        normModel = (models, None, "Norm")
        self.modelEval(normNoisedModel, reEval=False)
        self.modelEval(normModel, reEval=False)
        self.doPlot("Eps", [bestModel, "Norm",
                    "Norm_Noised", "Checksum"])

    def showNumericalFigure(self, best):
        """ Showing numerical results Of Norm, Norm_Noised and top  performing models"""
        bestModels = self.findBest(best)
        models = bestModels + ["Norm", "Norm_Noised"]
        for modelSaveName in models:
            epsRecord = load(
                f"{self.savePath}{modelSaveName}/Eps")
            print(modelSaveName)
            print(epsRecord.groupby("Noise")["Eps"].mean().tolist())

    def evaluate(self, reEval=False):
        for model in self.modelToEvaluate:
            self.modelEval(model, reEval)
