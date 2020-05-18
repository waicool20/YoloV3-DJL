package com.waicool20.djl.yolo

import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.translator.ObjectDetectionTranslator
import ai.djl.ndarray.NDList
import ai.djl.translate.Pipeline
import ai.djl.translate.TranslatorContext

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

        for(i in 2..4) {
            val output = list[i]
        }
        TODO()
    }
}