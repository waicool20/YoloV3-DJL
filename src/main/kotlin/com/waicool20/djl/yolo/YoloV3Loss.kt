package com.waicool20.djl.yolo

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.loss.AbstractCompositeLoss
import ai.djl.training.loss.Loss
import ai.djl.util.Pair
import com.waicool20.djl.util.*
import com.waicool20.djl.util.YoloUtils.bboxIOUs
import com.waicool20.djl.util.YoloUtils.whIOU

class YoloV3Loss(
    val ignoreThreshold: Double = 0.5,
    val lambdaCoord: Double = 5.0,
    val lambdaNoObj: Double = 0.5,
    val lossType: Type = Type.STANDARD,
    val focalAlpha: Double = 0.5,
    val focalGamma: Double = 2.0
) : AbstractCompositeLoss("YoloV3Loss") {
    enum class Type {
        STANDARD, IOU, GIOU, DIOU, CIOU
    }

    private inner class YoloV3LayerLoss(private val layerIndex: Int) : Loss("YoloV3LayerLoss") {
        override fun evaluate(labels: NDList, predictions: NDList): NDArray {
            val label = labels.singletonOrThrow().reshape(labels[0].shape[0], 5)
            val prediction = predictions[layerIndex + 2]

            val anchors = predictions[1][layerIndex.toLong()].reshape(3, 2)
            val batches = prediction.shape[0]
            val nAnchors = prediction.shape[1]
            val stride = prediction.shape[2]
            val nClasses = prediction.shape[4] - 5

            val trueXY = label.get(NDIndex("..., 1:3"))
            val trueWH = label.get(NDIndex("..., 3:5"))
            val trueObj = run {
                val x = (trueXY.get("..., 0") * stride).floor().toFloatArray()
                val y = (trueXY.get("..., 1") * stride).floor().toFloatArray()
                val ious = whIOU(trueWH, anchors)
                val n = ious.argMax(1).toLongArray()
                manager.zeros(Shape(batches, nAnchors, stride, stride)).apply {
                    for (b in 0 until ious.shape[0].toInt()) {
                        set(NDIndex(b.toLong(), n[b], x[b].toLong(), y[b].toLong()), 1)
                    }
                }
            }
            val trueObjMask = trueObj.toType(DataType.BOOLEAN, false)
            val trueCls = manager.zeros(Shape(batches, nClasses)).apply {
                val trueClsIndex = label.get(NDIndex("..., 0")).toFloatArray()
                for (b in 0 until batches) {
                    set(NDIndex(b, trueClsIndex[b.toInt()].toLong()), 1)
                }
            }

            val predXY = prediction.get(NDIndex("..., 0:2")).booleanMask(trueObjMask)
            val predWH = prediction.get(NDIndex("..., 2:4")).booleanMask(trueObjMask)
            val predObj = prediction.get(NDIndex("..., 4"))
            val predCls = prediction.get(NDIndex("..., 5:")).booleanMask(trueObjMask)

            val boxes1 = predXY.concat(predWH, 1)
            val boxes2 = trueXY.concat(trueWH, 1)
            val ious = when (lossType) {
                Type.STANDARD, Type.IOU -> bboxIOUs(boxes1, boxes2, YoloUtils.IOUType.IOU)
                Type.GIOU -> bboxIOUs(boxes1, boxes2, YoloUtils.IOUType.GIOU)
                Type.DIOU -> bboxIOUs(boxes1, boxes2, YoloUtils.IOUType.DIOU)
                Type.CIOU -> bboxIOUs(boxes1, boxes2, YoloUtils.IOUType.CIOU)
            }

            val weight = 2 - (trueWH.get(NDIndex(":, 0")) * trueWH.get(NDIndex(":, 1")))

            val bce = sigmoidBinaryCrossEntropyLoss("BCE")

            val xyLoss = when (lossType) {
                Type.STANDARD -> ((trueXY - predXY).square() * weight * lambdaCoord).mean()
                else -> (1 - ious).mean() * weight * lambdaCoord
            }
            val whLoss = when (lossType) {
                Type.STANDARD -> ((trueWH.sqrt() - predWH.sqrt()).square() * weight * lambdaCoord).mean()
                else -> null
            }

            val trueObjMasked = trueObj.booleanMask(trueObjMask)
            val predObjMasked = predObj.booleanMask(trueObjMask)

            val objLoss = bce.evaluate(NDList(trueObjMasked), NDList(predObjMasked))

            val noObjLoss = run {
                val ignoreMask = ious.lt(ignoreThreshold)
                val trueNoObjMask = trueObjMask.logicalNot()

                val trueNoObj = trueObj.booleanMask(trueNoObjMask)
                val predNoObj = predObj.booleanMask(trueNoObjMask)

                val predExtraNoObj = predObjMasked.booleanMask(ignoreMask)
                val trueExtraNoObj = trueObjMasked.booleanMask(ignoreMask)

                bce.evaluate(
                    NDList(trueNoObj.concat(trueExtraNoObj)),
                    NDList(predNoObj.concat(predExtraNoObj))
                ) * lambdaNoObj
            }

            val classLoss = bce.evaluate(NDList(trueCls), NDList(predCls))

            var totalLoss = xyLoss + objLoss + noObjLoss + classLoss
            if (whLoss != null) totalLoss = totalLoss + whLoss
            return totalLoss
        }
    }

    private lateinit var manager: NDManager

    init {
        components = List(3) { i -> YoloV3LayerLoss(i) }
    }

    override fun inputForComponent(componentIndex: Int, labels: NDList, predictions: NDList): Pair<NDList, NDList> {
        manager = predictions[0].manager.newSubManager()

        return when (componentIndex) {
            0 -> Pair(labels, predictions)
            1 -> Pair(labels, predictions)
            2 -> Pair(labels, predictions)
            else -> error("Invalid component index: $componentIndex")
        }
    }
}