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
import com.waicool20.djl.util.YoloUtils
import com.waicool20.djl.util.YoloUtils.bboxIOUs
import com.waicool20.djl.util.YoloUtils.whIOU
import com.waicool20.djl.util.fEllipsis
import com.waicool20.djl.util.plus
import com.waicool20.djl.util.times

class YoloV3Loss(
    val ignoreThreshold: Double = 0.5,
    val lambdaCoord: Double = 5.0,
    val lambdaNoObj: Double = 0.5,
    val lossType: Type = Type.STANDARD
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

            val trueXY = label.get(NDIndex(label.fEllipsis() + "1:3"))
            val trueWH = label.get(NDIndex(label.fEllipsis() + "3:5"))
            val trueObj = run {
                val x = (trueXY.get(trueXY.fEllipsis() + 0) * stride).floor()
                val y = (trueXY.get(trueXY.fEllipsis() + 1) * stride).floor()
                val ious = whIOU(trueWH, anchors)
                val n = ious.argMax(1)
                manager.zeros(Shape(batches, nAnchors, stride, stride)).apply {
                    for (b in 0 until ious.shape[0]) {
                        set(NDIndex(b, n.getLong(b), x.getFloat(b).toLong(), y.getFloat(b).toLong()), 1)
                    }
                }
            }
            val trueObjMask = trueObj.toType(DataType.BOOLEAN, false)
            val trueCls = manager.zeros(Shape(batches, nClasses)).apply {
                val trueClsIndex = label.get(NDIndex(label.fEllipsis() + "0"))
                for (b in 0 until batches) {
                    val i = trueClsIndex.getFloat(b).toLong()
                    set(NDIndex(b, i), 1)
                }
            }

            val predXY = prediction.get(NDIndex(prediction.fEllipsis() + "0:2")).booleanMask(trueObjMask)
            val predWH = prediction.get(NDIndex(prediction.fEllipsis() + "2:4")).booleanMask(trueObjMask)
            val predObj = prediction.get(NDIndex(prediction.fEllipsis() + "4"))
            val predCls = prediction.get(NDIndex(prediction.fEllipsis() + "5:")).booleanMask(trueObjMask)

            val boxes1 = predXY.concat(predWH, 1)
            val boxes2 = trueXY.concat(trueWH, 1)
            val ious = bboxIOUs(boxes1, boxes2)

            val weight = (trueWH.get(NDIndex(":, 0")) * trueWH.get(NDIndex(":, 1")).neg() + 2).mean().getFloat()

            val mse = l2Loss("MSE", weight * lambdaCoord.toFloat())
            val bce = sigmoidBinaryCrossEntropyLoss("BCE")

            val xyLoss = when (lossType) {
                Type.STANDARD -> mse.evaluate(NDList(trueXY), NDList(predXY))
                Type.IOU -> ious.mean() * lambdaCoord * weight
                Type.GIOU -> (bboxIOUs(boxes1, boxes2, YoloUtils.IOUType.GIOU).neg() + 1).mean() * lambdaCoord * weight
                Type.DIOU -> (bboxIOUs(boxes1, boxes2, YoloUtils.IOUType.DIOU).neg() + 1).mean() * lambdaCoord * weight
                Type.CIOU -> (bboxIOUs(boxes1, boxes2, YoloUtils.IOUType.CIOU).neg() + 1).mean() * lambdaCoord * weight
            }
            val whLoss = mse.evaluate(NDList(trueWH.sqrt()), NDList(predWH.sqrt()))
            val objLoss = bce.evaluate(
                NDList(trueObj.booleanMask(trueObjMask)),
                NDList(predObj.booleanMask(trueObjMask))
            )

            val noObjLoss = run {
                val ignoreMask = ious.lt(ignoreThreshold)
                val trueNoObjMask = trueObjMask.logicalNot()

                val trueNoObj = trueObj.booleanMask(trueNoObjMask)
                val predNoObj = predObj.booleanMask(trueNoObjMask)

                val trueExtraNoObj = trueObj.booleanMask(trueObjMask).booleanMask(ignoreMask)
                val predExtraNoObj = predObj.booleanMask(trueObjMask).booleanMask(ignoreMask)

                bce.evaluate(
                    NDList(trueNoObj.concat(trueExtraNoObj)),
                    NDList(predNoObj.concat(predExtraNoObj))
                ) * lambdaNoObj
            }

            val classLoss = bce.evaluate(NDList(trueCls), NDList(predCls))

            val totalLoss = xyLoss + whLoss + objLoss + noObjLoss + classLoss
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