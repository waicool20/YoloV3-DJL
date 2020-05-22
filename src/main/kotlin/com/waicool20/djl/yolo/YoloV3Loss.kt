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
import com.waicool20.djl.util.YoloUtils.bboxIOUs
import com.waicool20.djl.util.YoloUtils.whIOU
import com.waicool20.djl.util.fEllipsis
import com.waicool20.djl.util.minus
import com.waicool20.djl.util.plus
import com.waicool20.djl.util.times

class YoloV3Loss(
    val ignoreThreshold: Double = 0.5,
    val lambdaCoord: Double = 5.0,
    val lambdaNoObj: Double = 0.5
) : AbstractCompositeLoss("YoloV3Loss") {

    private inner class YoloV3LayerLoss(private val layerIndex: Int) : Loss("YoloV3LayerLoss") {
        override fun evaluate(labels: NDList, predictions: NDList): NDArray {
            val label = labels.singletonOrThrow().reshape(labels[0].shape[0], 5)
            val prediction = predictions[layerIndex + 2]

            val anchors = predictions[1][layerIndex.toLong()].reshape(3, 2)
            val stride = prediction.shape[2]

            val predXY = prediction.get(NDIndex(prediction.fEllipsis() + "0:2"))
            val predWH = prediction.get(NDIndex(prediction.fEllipsis() + "2:4"))
            val predObj = prediction.get(NDIndex(prediction.fEllipsis() + "4"))
            val predCls = prediction.get(NDIndex(prediction.fEllipsis() + "5:"))

            val trueXY = label.get(NDIndex(label.fEllipsis() + "1:3"))
            val trueWH = label.get(NDIndex(label.fEllipsis() + "3:5"))
            val trueObj = run {
                val x = (trueXY.get(trueXY.fEllipsis() + 0) * stride).floor()
                val y = (trueXY.get(trueXY.fEllipsis() + 1) * stride).floor()
                val ious = whIOU(trueWH, anchors)
                val n = ious.argMax(1)
                manager.zeros(predObj.shape).apply {
                    for (b in 0 until ious.shape[0]) {
                        set(NDIndex(b, n.getLong(b), x.getFloat(b).toLong(), y.getFloat(b).toLong()), 1)
                    }
                }.toType(DataType.BOOLEAN, false)
            }
            val trueCls = manager.zeros(Shape(predCls.shape[0], predCls.shape[4])).apply {
                val trueClsIndex = label.get(NDIndex(label.fEllipsis() + "0"))
                for (b in 0 until predCls.shape[0]) {
                    val i = trueClsIndex.getFloat(b).toLong()
                    set(NDIndex(b, i), 1)
                }
            }

            val weight = (trueWH.get(NDIndex(":, 0")) * trueWH.get(NDIndex(":, 1")) * -1 + 2).mean().getFloat()

            val mse = l2Loss("MSE", weight * lambdaCoord.toFloat())
            val bce = sigmoidBinaryCrossEntropyLoss("BCE")

            val xyLoss = run {
                val xy = predXY.booleanMask(trueObj)
                mse.evaluate(NDList(trueXY), NDList(xy))
            }
            val whLoss = run {
                val wh = predWH.booleanMask(trueObj).sqrt()
                val twh = trueWH.sqrt()
                mse.evaluate(NDList(twh), NDList(wh))
            }
            val objLoss = bce.evaluate(
                NDList(trueObj.toType(DataType.FLOAT32, true).booleanMask(trueObj)),
                NDList(predObj.booleanMask(trueObj))
            )

            val noObjLoss = run {
                val boxes1 = predXY.concat(predWH, 4).booleanMask(trueObj)
                val boxes2 = trueXY.concat(trueWH, 1)
                val ious = bboxIOUs(boxes1, boxes2)
                val ignoreMask = ious.lt(ignoreThreshold)
                val trueNoObjMask = trueObj.logicalNot()

                val trueNoObj = trueObj.toType(DataType.FLOAT32, true).booleanMask(trueNoObjMask)
                val predNoObj = predObj.booleanMask(trueNoObjMask)

                val trueExtraNoObj = trueObj.toType(DataType.FLOAT32, true).booleanMask(trueObj).booleanMask(ignoreMask)
                val predExtraNoObj = predObj.booleanMask(trueObj).booleanMask(ignoreMask)

                bce.evaluate(
                    NDList(trueNoObj.concat(trueExtraNoObj)),
                    NDList(predNoObj.concat(predExtraNoObj))
                ) * lambdaNoObj
            }

            val classLoss = bce.evaluate(
                NDList(trueCls),
                NDList(predCls.booleanMask(trueObj))
            )

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