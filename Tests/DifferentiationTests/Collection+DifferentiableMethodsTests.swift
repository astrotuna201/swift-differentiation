import Differentiation
import Testing

@Suite("Array Differentiable Methods Tests")
struct CollectionDifferentiableMethodsTests {
    @differentiable(reverse)
    func run1(arr: [Double], subrange: Range<Int>, with newValues: [Double]) -> [Double] {
        var arr = arr
        arr.replaceSubrange(subrange, with: newValues)
        return arr
    }

    @differentiable(reverse)
    func run2(arr: [Double], subrange: Range<Int>, with newValues: [Double]) -> [Double] {
        var arr = arr

        let eraseCount = subrange.count
        let insertCount = withoutDerivative(at: newValues.count)
        let growth = insertCount - eraseCount
        for (subrangeIndex, newValuesIndex) in zip(subrange, 0 ..< withoutDerivative(at: newValues.count)) {
            arr[cta: subrangeIndex] = newValues[newValuesIndex]
        }

        if growth > 0 {
            for _ in 0 ..< growth {
                arr.append(.zero)
            }

            for i in 0 ..< growth {
                let newValue = newValues[i + eraseCount]
                let offsetIndex = i + subrange.upperBound
                let oldValue = arr[offsetIndex]
                arr[cta: offsetIndex] = newValue
                arr[cta: offsetIndex + growth] = oldValue
            }
            return arr
        }
        else if growth < 0 {
            for i in 0 ..< abs(growth) {
                let index = i + subrange.lowerBound + insertCount
                arr[cta: index] = arr[i + subrange.upperBound]
            }
            let arrayCount = withoutDerivative(at: arr.count)
            var newArr = Array<Double>(repeating: .zero, count: arrayCount + growth)
            for i in 0 ..< arrayCount + growth {
                newArr[cta: i] = arr[i]
            }
            return newArr
        }
        else {
            return arr
        }
    }

    @Test
    func sliceArray() {
        @differentiable(reverse)
        func run(arr: [Double], range: Range<Int>) -> Double {
            arr[range].differentiableReduce(0, +)
        }

        let arr: [Double] = [100, 2, 3, 4, 100]

        let (value, pullback) = valueWithPullback(at: arr) { array in
            run(arr: array, range: 1 ..< 4)
        }

        #expect(value == 9)
        let gradient = pullback(1)
        #expect(gradient == [0, 1, 1, 1, 0])
    }

    @Test
    func rangeGetPullbackWithEmptyZeroCotangent() {
        // The range-get pullback scatters a slice cotangent into a
        // `forwardCount`-element zero buffer. An empty slice cotangent is
        // the canonical zero of every count; the scatter must preserve the
        // source array's count (replacing `bounds` with an empty
        // collection used to delete the slots and shrink the buffer).
        let arr: [Double] = [1, 2, 3, 4, 5, 6, 7, 8]
        let (_, scatter) = arr._vjpSubscriptRangeGet(bounds: 4 ..< 8)

        let zero = scatter(ArraySlice<Double>.DifferentiableView.zero)
        #expect(zero.base.count == 8)
        #expect(zero.base.allSatisfy { $0 == 0 })

        let live = scatter(ArraySlice<Double>.DifferentiableView([1, 2, 3, 4]))
        #expect(live.base.count == 8)
        #expect(live.base == [0, 0, 0, 0, 1, 2, 3, 4])
    }

    @Test
    func deadSliceWritebackScatterKeepsSourceCount() {
        // Two 2n->n size-changing slice writebacks into the fields of a
        // Differentiable struct, with the loss reading only ONE of the
        // written fields (the other writeback is a dead zero-cotangent
        // copy). Before the fix the dead writeback's scatter returned a
        // slice-count-sized (n) buffer instead of a source-count-sized
        // (2n) buffer, and the accumulation into the shared tangent tripped
        // the DifferentiableView.+ count precondition.
        @differentiable(reverse, wrt: x)
        func core(x: TwoFieldState, n: Int) -> TwoFieldState {
            var big = [Double](repeating: 0, count: 2 * n)
            for o in 0 ..< n {
                big[cta: o] = x.f0[o]
                big[cta: n + o] = x.f1[o]
            }
            var out = x
            out.f0 = Array(big[0 ..< n])
            out.f1 = Array(big[n ..< 2 * n])
            return out
        }

        let n = 4
        let x0 = TwoFieldState(f0: [0.3, 0.2, 0.3, 0.4], f1: [0.5, 0.6, 0.7, 0.8])
        let (_, pullback) = valueWithPullback(at: x0) { a in
            let c = core(x: a, n: n)
            var m: Double = 0
            for i in 0 ..< n where i % 3 != 0 {
                m += c.f0[i]
            }
            return m
        }
        let grad = pullback(1)
        #expect(grad.f0.base == [0, 1, 1, 0])
        #expect(grad.f1.base.allSatisfy { $0 == 0 })
    }

    @Test
    func testReplacingSubrange() {
        let a: [Double] = [0, 1, 2, 3, 4]
        let replacement: [Double] = [10, 11]
        let subrange: Range<Int> = 2 ..< 4

        #expect(run1(arr: a, subrange: subrange, with: replacement) == [0, 1, 10, 11, 4])
        #expect(run2(arr: a, subrange: subrange, with: replacement) == [0, 1, 10, 11, 4])

        let (value1, pullback1) = valueWithPullback(at: a, replacement, of: { a, replacement in
            run1(arr: a, subrange: subrange, with: replacement)
        })

        let (value2, pullback2) = valueWithPullback(at: a, replacement, of: { a, replacement in
            run2(arr: a, subrange: subrange, with: replacement)
        })

        #expect(value1 == value2)

        let gradient1 = pullback1([1, 0, 0, 0, 0])
        let gradient2 = pullback2([1, 0, 0, 0, 0])

        #expect(gradient1 == gradient2)

        let gradient3 = pullback1([0, 0, 1, 0, 0])
        let gradient4 = pullback2([0, 0, 1, 0, 0])

        #expect(gradient3 == gradient4)
    }

    @Test
    func testReplacingWithLongerSubrange() {
        let a: [Double] = [0, 1, 2, 3, 4]
        let replacement: [Double] = [10, 11, 22]
        let subrange: Range<Int> = 2 ..< 4

        #expect(run1(arr: a, subrange: subrange, with: replacement) == [0, 1, 10, 11, 22, 4])
        #expect(run2(arr: a, subrange: subrange, with: replacement) == [0, 1, 10, 11, 22, 4])

        let (value1, pullback1) = valueWithPullback(at: a, replacement, of: { a, replacement in
            run1(arr: a, subrange: subrange, with: replacement)
        })

        let (value2, pullback2) = valueWithPullback(at: a, replacement, of: { a, replacement in
            run2(arr: a, subrange: subrange, with: replacement)
        })

        #expect(value1 == value2)

        let gradient1 = pullback1([1, 0, 0, 0, 0, 0])
        let gradient2 = pullback2([1, 0, 0, 0, 0, 0])

        #expect(gradient1 == gradient2)

        let gradient3 = pullback1([0, 0, 1, 0, 0, 0])
        let gradient4 = pullback2([0, 0, 1, 0, 0, 0])

        #expect(gradient3 == gradient4)
    }

    @Test
    func testReplacingWithShorterSubrange() {
        let a: [Double] = [0, 1, 2, 3, 4]
        let replacement: [Double] = [10]
        let subrange: Range<Int> = 2 ..< 4

        #expect(run1(arr: a, subrange: subrange, with: replacement) == [0, 1, 10, 4])
        #expect(run2(arr: a, subrange: subrange, with: replacement) == [0, 1, 10, 4])

        let (value1, pullback1) = valueWithPullback(at: a, replacement, of: { a, replacement in
            run1(arr: a, subrange: subrange, with: replacement)
        })

        let (value2, pullback2) = valueWithPullback(at: a, replacement, of: { a, replacement in
            run2(arr: a, subrange: subrange, with: replacement)
        })

        #expect(value1 == value2)

        let gradient1 = pullback1([1, 0, 0, 0])
        let gradient2 = pullback2([1, 0, 0, 0])

        #expect(gradient1 == gradient2)

        let gradient3 = pullback1([0, 0, 1, 0])
        let gradient4 = pullback2([0, 0, 1, 0])

        #expect(gradient3 == gradient4)
    }
}

struct TwoFieldState: Differentiable {
    var f0: [Double]
    var f1: [Double]
}
