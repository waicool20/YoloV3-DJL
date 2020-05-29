package com.waicool20.djl.yolo

import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.output.Rectangle
import ai.djl.modality.cv.translator.ObjectDetectionTranslator
import ai.djl.ndarray.NDList
import ai.djl.ndarray.index.NDIndex
import ai.djl.translate.Pipeline
import ai.djl.translate.TranslatorContext
import com.waicool20.djl.util.YoloUtils
import com.waicool20.djl.util.minus

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
        .setClasses(classes)
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

            val w = output.get(NDIndex(":, 2"))
            val h = output.get(NDIndex(":, 3"))

            val x = output.get(NDIndex(":, 0")) - w / 2
            val y = output.get(NDIndex(":, 1")) - h / 2

            val p = output.get(NDIndex(":, 4"))
            val c = output.get(NDIndex(":, 5:")).argMax(1)

            for (j in 0 until output.shape[0]) {
                val probability = p.getFloat(j).toDouble()
                if (probability >= threshold) {
                    val ox = x.getFloat(j).toDouble()
                    if (ox !in 0.0..1.0) continue
                    val oy = y.getFloat(j).toDouble()
                    if (oy !in 0.0..1.0) continue
                    val ow = w.getFloat(j).toDouble()
                    if (ow !in 0.0..1.0) continue
                    val oh = h.getFloat(j).toDouble()
                    if (oh !in 0.0..1.0) continue
                    objects.add(
                        DetectedObjects.DetectedObject(
                            classes[c.getLong(j).toInt()],
                            probability,
                            Rectangle(ox, oy, ow, oh)
                        )
                    )
                }
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