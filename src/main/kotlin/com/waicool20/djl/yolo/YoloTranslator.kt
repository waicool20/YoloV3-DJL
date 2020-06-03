package com.waicool20.djl.yolo

import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.output.Rectangle
import ai.djl.modality.cv.translator.ObjectDetectionTranslator
import ai.djl.ndarray.NDList
import ai.djl.ndarray.index.NDIndex
import ai.djl.translate.Pipeline
import ai.djl.translate.TranslatorContext
import com.waicool20.djl.util.YoloUtils

class YoloTranslator(
    pipeline: Pipeline,
    val iouThreshold: Double = 0.5,
    val threshold: Double = 0.2,
    val classes: List<String> = emptyList(),
    val rescaleSize: Pair<Double, Double> = 0.0 to 0.0
) : ObjectDetectionTranslator(
    Builder()
        .setPipeline(pipeline)
        .optThreshold(threshold.toFloat())
        .optSynset(classes)
        .optRescaleSize(rescaleSize.first, rescaleSize.second)
) {
    private class Builder : ObjectDetectionTranslator.BaseBuilder<Builder>() {
        override fun self() = this
    }

    override fun processOutput(ctx: TranslatorContext, list: NDList): DetectedObjects {
        val objects = mutableListOf<DetectedObjects.DetectedObject>()
        for (i in 0 until 3) {
            var output = list[i]
            val shape = output.shape
            output = output.reshape(shape[0] * shape[1] * shape[2], shape[3])
            output.set(NDIndex( ":, 4:")) { it.ndArrayInternal.sigmoid() }
            output = output.booleanMask(output.get(NDIndex(":, 4")).gte(threshold))

            for (j in 0 until output.shape[0]) {
                val detection = output[j].toFloatArray()
                val ow = detection[2].toDouble()
                val oh = detection[3].toDouble()
                val ox = detection[0] - ow / 2
                val oy = detection[1] - oh / 2
                if (ox !in 0.0..1.0) continue
                if (oy !in 0.0..1.0) continue
                if (ow !in 0.0..1.0) continue
                if (oh !in 0.0..1.0) continue

                val p = detection[4].toDouble()
                val c = detection.slice(5..detection.lastIndex)
                val cMaxIdx = c.indexOf(c.max())
                objects.add(DetectedObjects.DetectedObject(classes[cMaxIdx], p, Rectangle(ox, oy, ow, oh)))
            }
        }
        val keep = YoloUtils.nms(iouThreshold, objects)
        return DetectedObjects(
            keep.map { it.className },
            keep.map { it.probability },
            keep.map { it.boundingBox }
        )
    }
}