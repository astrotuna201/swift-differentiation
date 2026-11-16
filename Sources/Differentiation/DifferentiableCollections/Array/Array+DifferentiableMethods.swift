import _Differentiation

extension Array where Element: Differentiable {
    @derivative(of: init)
    @inlinable
    static func _vjpInit<C: Collection>(_ c: C) -> (
        value: Self,
        pullback: (Self.TangentVector) -> C.TangentVector
    ) where
        C: Differentiable,
        C.Element == Element,
        C.TangentVector: RangeReplaceableCollection,
        Element.TangentVector == C.TangentVector.Element
    {
        (
            value: .init(c),
            pullback: { v in
                C.TangentVector(v)
            }
        )
    }
}

extension Array where Element: Differentiable {
    // TODO: Make this work more generally
    @derivative(of: Array.subscript.get)
    @inlinable
    public func _vjpSubscriptRangeGet(bounds: Range<Int>) -> (
        value: ArraySlice<Element>,
        pullback: (ArraySlice<Element>.TangentVector) -> Array<Element>.TangentVector
    ) {
        let forwardCount = self.count
        return (
            value: self[bounds],
            pullback: { v in
                var result = Array<Element>.TangentVector(repeating: .zero, count: forwardCount)
                // An empty `v` is the canonical zero cotangent of every count
                // (see `DifferentiableView.zero`); replacing `bounds` with an
                // empty collection would delete those slots and shrink the
                // `forwardCount`-element buffer, so the zero scatter must be
                // a no-op on the buffer.
                if !v.base.isEmpty {
                    result.replaceSubrange(bounds, with: v)
                }
                return result
            }
        )
    }
}

extension Array where Element: Differentiable {
    // TODO: Can we make this work for all RangeReplaceableCollections?
    @derivative(of: replaceSubrange)
    @inlinable
    public mutating func _vjpReplaceSubrange<C>(
        _ subrange: Range<Self.Index>,
        with newElements: C
    ) -> (
        value: (),
        pullback: (inout Self.TangentVector) -> C.TangentVector
    ) where C: Collection, C: Differentiable, Element == C.Element, C.TangentVector: RangeReplaceableCollection,
        C.TangentVector.Element == C.Element.TangentVector
    {
        let lowerBound = subrange.lowerBound
        let newElementCount = newElements.count
        let subrangeCount = subrange.count
        self.replaceSubrange(subrange, with: newElements)
        return (
            value: (),
            pullback: { v in
                // An empty `v` is the canonical zero cotangent of every
                // count (see `DifferentiableView.zero`); the new-elements
                // region is then zero, and the zero self cotangent needs no
                // layout restoration.
                if v.base.isEmpty {
                    return C.TangentVector()
                }
                let newElementsRange = lowerBound ..< lowerBound + newElementCount
                let dElement = v[newElementsRange]
                v.replaceSubrange(newElementsRange, with: repeatElement(.zero, count: subrangeCount))
                return C.TangentVector(dElement)
            }
        )
    }
}
