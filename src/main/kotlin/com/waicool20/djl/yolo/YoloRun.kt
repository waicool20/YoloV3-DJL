package com.waicool20.djl.yolo

import ai.djl.Device
import ai.djl.Model
import ai.djl.basicdataset.PikachuDetection
import ai.djl.metric.Metrics
import ai.djl.modality.cv.ImageVisualization
import ai.djl.modality.cv.transform.Resize
import ai.djl.modality.cv.transform.ToTensor
import ai.djl.modality.cv.util.BufferedImageUtils
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.TrainingConfig
import ai.djl.training.dataset.Dataset
import ai.djl.training.dataset.RandomAccessDataset
import ai.djl.training.listener.TrainingListener
import ai.djl.training.optimizer.Adam
import ai.djl.training.optimizer.learningrate.LearningRateTracker
import ai.djl.training.util.ProgressBar
import ai.djl.translate.Pipeline
import com.waicool20.djl.util.SequentialBlock
import com.waicool20.djl.util.TopLeftXYToCenterXY
import com.waicool20.djl.util.XYXYToXYWH
import com.waicool20.djl.util.openWindowPreview
import java.awt.GraphicsEnvironment
import java.nio.file.Paths
import javax.imageio.ImageIO

private val BATCH_SIZE = 1
private val WIDTH = 416
private val HEIGHT = 416
private val EPOCH = 32

private val pipeline = Pipeline(Resize(WIDTH, HEIGHT), ToTensor())

fun main() {
    //predictYolo()
    trainYolo()
}

private fun predictYolo() {
    val yolov3 = YoloV3(numClasses = 1)
    val model = Model.newInstance()
    model.block = yolov3
    model.load(Paths.get(""), "yolov3")

    val predictBlock = SequentialBlock {
        add(model.block)
        add { NDList(it[2], it[3], it[4]) }
    }
    model.block = predictBlock

    val translator = YoloTranslator(
        iouThreshold = 0.2,
        threshold = 1.0,
        pipeline = pipeline,
        classes = listOf("Pikachu")
    )

    val predictor = model.newPredictor(translator)
    val image = BufferedImageUtils.fromFile(Paths.get("test.jpg"))
    val objects = predictor.predict(image)
    ImageVisualization.drawBoundingBoxes(image, objects)
    if (!GraphicsEnvironment.isHeadless()) image.openWindowPreview()
    val out = Paths.get("").resolve("output.png")
    ImageIO.write(image, "png", out.toFile())
}

private fun trainYolo() {
    val trainDataset = getDataset(Dataset.Usage.TRAIN)
    val testDataset = getDataset(Dataset.Usage.TEST)

    val yolov3 = YoloV3(numClasses = 1)
    val model = Model.newInstance()
    model.block = yolov3

    var lastEpoch = 0
    try {
        model.load(Paths.get(""), "yolov3")
        lastEpoch = model.getProperty("Epoch").toInt()
    } catch (e: Exception) {
        println("No weights found, training new weights")
    }

    val trainer = model.newTrainer(getTrainingConfig())
    trainer.metrics = Metrics()
    val inputShape = Shape(BATCH_SIZE.toLong(), 3, WIDTH.toLong(), HEIGHT.toLong())
    trainer.initialize(inputShape)

    for (epoch in lastEpoch until (lastEpoch + EPOCH)) {
        for (batch in trainer.iterateDataset(trainDataset)) {
            trainer.trainBatch(batch)
            trainer.step()
            batch.close()
            System.gc()
        }
        for (batch in trainer.iterateDataset(testDataset)) {
            trainer.validateBatch(batch)
            batch.close()
            System.gc()
        }

        trainer.endEpoch()
        trainer.model.apply {
            setProperty("Epoch", epoch.toString())
            save(Paths.get(""), "yolov3")
        }
    }
    model.save(Paths.get(""), "yolov3")
}

private fun getDataset(usage: Dataset.Usage): RandomAccessDataset {
    val pikachuDetection: PikachuDetection = PikachuDetection.builder()
        .optUsage(usage)
        .optPipeline(pipeline)
        .optTargetPipeline(Pipeline(XYXYToXYWH(), TopLeftXYToCenterXY()))
        .setSampling(BATCH_SIZE, true)
        .build()
    pikachuDetection.prepare(ProgressBar())
    return pikachuDetection
}

private fun getTrainingConfig(): TrainingConfig {
    val optimizer = Adam.builder()
        .optLearningRateTracker(LearningRateTracker.fixedLearningRate(0.0001f))
        .build()
    return DefaultTrainingConfig(YoloV3Loss())
        .optOptimizer(optimizer)
        .optDevices(arrayOf(Device.gpu()))
        .addTrainingListeners(*TrainingListener.Defaults.logging())
}