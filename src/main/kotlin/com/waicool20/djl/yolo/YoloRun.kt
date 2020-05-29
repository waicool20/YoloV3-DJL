package com.waicool20.djl.yolo

import ai.djl.Device
import ai.djl.Model
import ai.djl.basicdataset.PikachuDetection
import ai.djl.metric.Metrics
import ai.djl.modality.cv.ImageFactory
import ai.djl.modality.cv.transform.Resize
import ai.djl.modality.cv.transform.ToTensor
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
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
import org.slf4j.LoggerFactory
import java.awt.GraphicsEnvironment
import java.awt.image.BufferedImage
import java.nio.file.Paths
import javax.imageio.ImageIO

private val MODEL_NAME = "yolov3"
private val WIDTH = 416
private val HEIGHT = 416

private val pipeline = Pipeline(Resize(WIDTH, HEIGHT), ToTensor())

// Training parameters

private val BATCH_SIZE = 2
private val EPOCH = 48
private val IGNORE_THRESHOLD = 0.5
private val LAMBDA_COORD = 5.0
private val LAMBDA_NOOBJ = 0.5
private val LEARNING_RATE = 1e-5f
private val LOSS_TYPE = YoloV3Loss.Type.STANDARD
private val FOCAL_ALPHA = 0.5
private val FOCAL_GAMMA = 2.0

// Predict parameters

private val IOU_THRESHOLD = 0.2
private val PREDICT_THRESHOLD = 0.99

private val logger = LoggerFactory.getLogger("YoloRun")

fun main(args: Array<String>) {
    if (args.contains("--predict")) {
        predictYolo()
    } else {
        trainYolo()
    }
}

private fun predictYolo() {
    val yolov3 = YoloV3(numClasses = 1)
    val model = Model.newInstance(MODEL_NAME)
    model.block = yolov3
    model.load(Paths.get(""), MODEL_NAME)

    val predictBlock = SequentialBlock {
        add(model.block)
        add { NDList(it[2], it[3], it[4]) }
    }
    model.block = predictBlock

    val translator = YoloTranslator(
        iouThreshold = IOU_THRESHOLD,
        threshold = PREDICT_THRESHOLD,
        pipeline = pipeline,
        classes = listOf("Pikachu")
    )

    val predictor = model.newPredictor(translator)
    val image = ImageFactory.getInstance().fromFile(Paths.get("test.jpg"))
    val objects = predictor.predict(image)
    logger.info(objects.toString())
    image.drawBoundingBoxes(objects)
    if (!GraphicsEnvironment.isHeadless()) image.openWindowPreview()
    val out = Paths.get("").resolve("output.png")
    ImageIO.write(image.wrappedImage as BufferedImage, "png", out.toFile())
}

private fun trainYolo() {
    val trainDataset = getDataset(Dataset.Usage.TRAIN)
    val testDataset = getDataset(Dataset.Usage.TEST)

    val yolov3 = YoloV3(numClasses = 1)
    val model = Model.newInstance(MODEL_NAME)
    model.block = yolov3

    var lastEpoch = 0
    try {
        model.load(Paths.get(""), MODEL_NAME)
        lastEpoch = model.getProperty("Epoch").toInt() + 1
    } catch (e: Exception) {
        logger.warn("No weights found, training new weights (${e.message})")
    }

    val trainer = model.newTrainer(getTrainingConfig())
    trainer.metrics = Metrics()
    val inputShape = Shape(BATCH_SIZE.toLong(), 3, WIDTH.toLong(), HEIGHT.toLong())
    trainer.initialize(inputShape)

    for (epoch in lastEpoch until EPOCH) {
        for (batch in trainer.iterateDataset(trainDataset)) {
            EasyTrain.trainBatch(trainer, batch)
            trainer.step()
            batch.close()
        }
        for (batch in trainer.iterateDataset(testDataset)) {
            EasyTrain.validateBatch(trainer, batch)
            batch.close()
        }

        trainer.notifyListeners { it.onEpoch(trainer) }
        trainer.model.apply {
            setProperty("Epoch", epoch.toString())
            save(Paths.get(""), MODEL_NAME)
        }
    }
    model.save(Paths.get(""), MODEL_NAME)
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
        .optLearningRateTracker(LearningRateTracker.fixedLearningRate(LEARNING_RATE))
        .build()
    val loss = YoloV3Loss(
        IGNORE_THRESHOLD,
        LAMBDA_COORD,
        LAMBDA_NOOBJ,
        LOSS_TYPE,
        FOCAL_ALPHA,
        FOCAL_GAMMA
    )
    return DefaultTrainingConfig(loss)
        .optOptimizer(optimizer)
        .optDevices(arrayOf(Device.gpu()))
        .addTrainingListeners(*TrainingListener.Defaults.logging("train-log"))
}