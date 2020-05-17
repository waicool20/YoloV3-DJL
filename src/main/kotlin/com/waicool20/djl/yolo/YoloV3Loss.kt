package com.waicool20.djl.yolo

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.loss.AbstractCompositeLoss
import ai.djl.training.loss.Loss
import ai.djl.util.Pair
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.roundToLong

class YoloV3Loss(
    val ignoreThreshold: Double = 0.5
) : AbstractCompositeLoss("YoloV3Loss") {

    private inner class YoloV3LayerLoss(private val layerIndex: Int) : AbstractCompositeLoss("YoloV3LayerLoss") {
        init {
            components = listOf(
                Loss.l2Loss("BoundingBoxXLoss"),
                Loss.l2Loss("BoundingBoxYLoss"),
                Loss.l2Loss("BoundingBoxWLoss"),
                Loss.l2Loss("BoundingBoxHLoss"),
                Loss.sigmoidBinaryCrossEntropyLoss("ConfObjLoss"),
                Loss.sigmoidBinaryCrossEntropyLoss("ConfNoObjLoss", 100f, false),
                Loss.sigmoidBinaryCrossEntropyLoss("ClassLoss")
            )
        }

        override fun inputForComponent(componentIndex: Int, labels: NDList, predictions: NDList): Pair<NDList, NDList> {
            val targets = buildTargets(labels, predictions)
            val output = predictions[layerIndex + 2]
            return when (componentIndex) {
                0 -> {
                    val mask = targets[2]
                    val tx = targets[4].muli(mask)
                    val x = output.get(NDIndex(":, :, :, :, 0")).reshape(mask.shape).mul(mask)
                    Pair(NDList(tx), NDList(x))
                }
                1 -> {
                    val mask = targets[2]
                    val ty = targets[5].muli(mask)
                    val y = output.get(NDIndex(":, :, :, :, 1")).reshape(mask.shape).mul(mask)
                    Pair(NDList(ty), NDList(y))
                }
                2 -> {
                    val mask = targets[2]
                    val tw = targets[6].muli(mask)
                    val w = output.get(NDIndex(":, :, :, :, 2")).reshape(mask.shape).mul(mask)
                    Pair(NDList(tw), NDList(w))
                }
                3 -> {
                    val mask = targets[2]
                    val th = targets[7].muli(mask)
                    val h = output.get(NDIndex(":, :, :, :, 3")).reshape(mask.shape).mul(mask)
                    Pair(NDList(th), NDList(h))
                }
                4 -> {
                    val mask = targets[2]
                    val tc = targets[9].muli(mask)
                    val c = output.get(NDIndex(":, :, :, :, 4")).reshape(mask.shape).mul(mask)
                    Pair(NDList(tc), NDList(c))
                }
                5 -> {
                    val mask = targets[3]
                    val tc = targets[9].muli(mask)
                    val c = output.get(NDIndex(":, :, :, :, 4")).reshape(mask.shape).mul(mask)
                    Pair(NDList(tc), NDList(c))
                }
                6 -> {
                    val mask = targets[2].expandDims(0).repeat(0, targets[8].shape[4])
                        .transpose(1, 2, 3, 4, 0)
                    val tcls = targets[8].muli(mask)
                    val cls = output.get(NDIndex(":, :, :, :, 5:")).reshape(mask.shape).mul(mask)
                    Pair(NDList(tcls), NDList(cls))
                }
                else -> error("Invalid component index: $componentIndex")
            }
        }

        /**
         * @see <a href="https://towardsdatascience.com/calculating-loss-of-yolo-v3-layer-8878bfaaf1ff">Calculating Loss</a>
         */
        private fun buildTargets(labels: NDList, predictions: NDList): NDList {
            val anchors = predictions[1][layerIndex.toLong()].reshape(3, 2)
            val output = predictions[layerIndex + 2]
            var target = labels.singletonOrThrow()
            target = target.reshape(target.shape[0], 5)

            val nB = output.shape[0]
            val nA = output.shape[1]
            val nC = output.shape[4] - 5
            val nG = output.shape[3]

            val maskShape = Shape(nB, nA, nG, nG)

            val objMask = manager.zeros(maskShape)
            val noObjMask = manager.ones(maskShape)
            val classMask = manager.zeros(maskShape)
            val iouScores = manager.zeros(maskShape)
            val tx = manager.zeros(maskShape)
            val ty = manager.zeros(maskShape)
            val tw = manager.zeros(maskShape)
            val th = manager.zeros(maskShape)
            val tcls = manager.zeros(maskShape.addAll(Shape(nC)))

            val targetBoxes = target.get(NDIndex(":, 1:5")).mul(nG)
            val gxy = targetBoxes.get(NDIndex(":, :2"))
            val gwh = targetBoxes.get(NDIndex(":, 2:"))

            val ious = NDArrays.concat(NDList(whIOU(anchors[0], gwh), whIOU(anchors[1], gwh), whIOU(anchors[2], gwh)))
                .reshape(nA, nB)
            val bestIOU = ious.max(intArrayOf(0))
            val bestIOUn = ious.argMax(0)

            for (b in 0 until nB) {
                val gx = gxy.getFloat(b, 0)
                val gy = gxy.getFloat(b, 1)
                val gi = gx.roundToLong()
                val gj = gy.roundToLong()

                objMask.set(NDIndex(b, bestIOUn.getLong(b), gj, gi), 1)
                noObjMask.set(NDIndex(b, bestIOUn.getLong(b), gj, gi), 0)
                tx.set(NDIndex(b, bestIOUn.getLong(b), gj, gi), gx - floor(gx))
                ty.set(NDIndex(b, bestIOUn.getLong(b), gj, gi), gy - floor(gy))

                tcls.set(NDIndex(b, bestIOUn.getLong(b), gj, gi, target.getFloat(b, 0).roundToLong()), 1)

                for (a in 0 until ious.shape[0]) {
                    if (ious.getFloat(a, b) > ignoreThreshold) {
                        noObjMask.set(NDIndex(b, a, gj, gi), 0)
                    }
                }
                if (output.get(NDIndex(b, bestIOUn.getLong(b), gj, gi)).argMax().getLong() == target.getFloat(b, 0)
                        .roundToLong()
                ) {
                    classMask.set(NDIndex(b, bestIOUn.getLong(b), gj, gi), 1f)
                }
                val bboxIOU = bboxIOU(
                    output[NDIndex(b, bestIOUn.getLong(b), gj, gi)],
                    target[NDIndex("$b, 1:5")]
                )
                iouScores.set(NDIndex(b, bestIOUn.getLong(b), gj, gi), bboxIOU)
            }
            return NDList(
                iouScores,
                classMask,
                objMask,
                noObjMask,
                tx, ty,
                tw, th, tcls, objMask.toType(DataType.FLOAT32, true)
            )
        }

        private fun whIOU(wh1: NDArray, wh2: NDArray): NDArray {
            val wh2 = wh2.transpose()
            val w1 = wh1[0]
            val h1 = wh1[1]
            val w2 = wh2[0]
            val h2 = wh2[1]

            val inter = NDArrays.minimum(w1, w2).mul(NDArrays.minimum(h1, h2))
            val union = w1.mul(h1).add(1e-16).add(w2.mul(h2)).sub(inter)
            return inter.div(union)
        }

        private fun bboxIOU(box1: NDArray, box2: NDArray): Float {
            val b1x1 = box1.getFloat(0) - box1.getFloat(2) / 2
            val b1x2 = box1.getFloat(0) + box1.getFloat(2) / 2
            val b1y1 = box1.getFloat(1) - box1.getFloat(3) / 2
            val b1y2 = box1.getFloat(1) + box1.getFloat(3) / 2

            val b2x1 = box2.getFloat(0) - box2.getFloat(2) / 2
            val b2x2 = box2.getFloat(0) + box2.getFloat(2) / 2
            val b2y1 = box2.getFloat(1) - box2.getFloat(3) / 2
            val b2y2 = box2.getFloat(1) + box2.getFloat(3) / 2

            val rx1 = max(b1x1, b2x1)
            val ry1 = max(b1y1, b2y1)
            val rx2 = max(b1x2, b2x2)
            val ry2 = max(b1y2, b2y2)

            val interArea = (rx1 - rx2) * (ry1 - ry2)

            val b1a = (b1x2 - b1x1) * (b1y2 - b1y1)
            val b2a = (b2x2 - b2x1) * (b2y2 - b2y1)
            return interArea / (b1a + b2a - interArea + 1e-16f)
        }
    }

    private lateinit var manager: NDManager

    init {
        components = List(3) { i -> YoloV3LayerLoss(i) }
    }

    override fun inputForComponent(componentIndex: Int, labels: NDList, predictions: NDList): Pair<NDList, NDList> {
        manager = predictions[0].manager

        return when (componentIndex) {
            0 -> Pair(labels, predictions)
            1 -> Pair(labels, predictions)
            2 -> Pair(labels, predictions)
            else -> error("Invalid component index: $componentIndex")
        }
    }
}