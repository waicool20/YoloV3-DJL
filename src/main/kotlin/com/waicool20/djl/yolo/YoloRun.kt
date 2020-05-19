package com.waicool20.djl.yolo

import ai.djl.Device
import ai.djl.Model
import ai.djl.basicdataset.PikachuDetection
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
import ai.djl.training.util.ProgressBar
import ai.djl.translate.Pipeline
import com.waicool20.djl.util.SequentialBlock
import com.waicool20.djl.util.TopLeftXYToCenterXY
import com.waicool20.djl.util.XYMinMaxToXYWH
import java.nio.file.Paths
import javax.imageio.ImageIO

private val BATCH_SIZE = 1
private val WIDTH = 416
private val HEIGHT = 416
private val EPOCH = 1

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
        pipeline = pipeline,
        classes = listOf("Pikachu")
    )

    val predictor = model.newPredictor(translator)
    val image = BufferedImageUtils.fromFile(Paths.get("test.jpg"))
    val objects = predictor.predict(image)
    ImageVisualization.drawBoundingBoxes(image, objects)
    val out = Paths.get("").resolve("output.png")
    ImageIO.write(image, "png", out.toFile())
}

private fun trainYolo() {
    val trainDataset = getDataset(Dataset.Usage.TRAIN)
    val testDataset = getDataset(Dataset.Usage.TEST)

    val yolov3 = YoloV3(numClasses = 1)
    val model = Model.newInstance()
    model.block = yolov3

    val trainer = model.newTrainer(getTrainingConfig())
    val inputShape = Shape(BATCH_SIZE.toLong(), 3, WIDTH.toLong(), HEIGHT.toLong())
    trainer.initialize(inputShape)

    for (epoch in 0 until EPOCH) {
        for (batch in trainer.iterateDataset(trainDataset)) {
            trainer.trainBatch(batch)
            trainer.step()
            batch.close()
        }
        for (batch in trainer.iterateDataset(testDataset)) {
            trainer.validateBatch(batch)
            batch.close()
        }

        trainer.endEpoch()
        trainer.model.apply {
            setProperty("Epoch", java.lang.String.valueOf(epoch))
            save(Paths.get(""), "yolov3")
        }
    }
    model.save(Paths.get(""), "yolov3")
}

private fun getDataset(usage: Dataset.Usage): RandomAccessDataset {
    val pikachuDetection: PikachuDetection = PikachuDetection.builder()
        .optUsage(usage)
        .optPipeline(pipeline)
        .optTargetPipeline(Pipeline(XYMinMaxToXYWH(), TopLeftXYToCenterXY()))
        .setSampling(BATCH_SIZE, true)
        .build()
    pikachuDetection.prepare(ProgressBar())
    return pikachuDetection
}

private fun getTrainingConfig(): TrainingConfig {
    return DefaultTrainingConfig(YoloV3Loss())
        .optDevices(arrayOf(Device.cpu()))
        .addTrainingListeners(*TrainingListener.Defaults.logging())
}