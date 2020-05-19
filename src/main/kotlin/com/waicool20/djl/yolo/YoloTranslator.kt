package com.waicool20.djl.yolo

import ai.djl.modality.cv.output.BoundingBox
import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.output.Rectangle
import ai.djl.modality.cv.translator.ObjectDetectionTranslator
import ai.djl.ndarray.NDList
import ai.djl.ndarray.index.NDIndex
import ai.djl.translate.Pipeline
import ai.djl.translate.TranslatorContext
import com.waicool20.djl.util.minus

class YoloTranslator(
    pipeline: Pipeline,
    val threshold: Float = 0.2f,
    val classes: List<String> = emptyList(),
    val rescaleSize: Pair<Double, Double> = 0.0 to 0.0
) : ObjectDetectionTranslator(
    Builder()
        .setPipeline(pipeline)
        .optThreshold(threshold)
        .setClasses(classes)
        .optRescaleSize(rescaleSize.first, rescaleSize.second)
) {
    private class Builder : ObjectDetectionTranslator.BaseBuilder<Builder>() {
        override fun self() = this
    }

    override fun processOutput(ctx: TranslatorContext, list: NDList): DetectedObjects {
        val names = mutableListOf<String>()
        val rects = mutableListOf<BoundingBox>()
        val probabilities = mutableListOf<Double>()
        for (i in 0 until 3) {
            var output = list[i]
            val shape = output.shape
            output = output.reshape(shape[0] * shape[1] * shape[2], shape[3])

            val w = output.get(NDIndex(":, 2"))
            val h = output.get(NDIndex(":, 3"))

            val x = output.get(NDIndex(":, 0")) - w / 2
            val y = output.get(NDIndex(":, 1")) - h / 2

            val p = output.get(NDIndex(":, 4"))
            val c = output.get(NDIndex(":, 5:")).argMax(1)

            for (j in 0 until output.shape[0]) {
                val probability = p.getFloat(j).toDouble()
                if (probability > threshold) {
                    probabilities.add(probability)
                    names.add(classes[c.getLong(j).toInt()])
                    rects.add(
                        Rectangle(
                            x.getFloat(j).toDouble(),
                            y.getFloat(j).toDouble(),
                            w.getFloat(j).toDouble(),
                            h.getFloat(j).toDouble()
                        )
                    )
                }
            }
        }
        // TODO NMS
        return DetectedObjects(names, probabilities, rects)
    }
}