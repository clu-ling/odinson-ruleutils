from __future__ import annotations

import re
import json
import itertools
from typing import Dict, List, Optional, Text, Tuple, Type, Union
from odinson.ruleutils import config


__all__ = [
    "AstNode",
    "Matcher",
    "HoleMatcher",
    "ExactMatcher",
    "Constraint",
    "HoleConstraint",
    "FieldConstraint",
    "OrConstraint",
    "AndConstraint",
    "NotConstraint",
    "Surface",
    "HoleSurface",
    "WildcardSurface",
    "TokenSurface",
    "ConcatSurface",
    "OrSurface",
    "RepeatSurface",
    "Traversal",
    "HoleTraversal",
    "IncomingLabelTraversal",
    "OutgoingLabelTraversal",
    "IncomingWildcardTraversal",
    "OutgoingWildcardTraversal",
    "ConcatTraversal",
    "OrTraversal",
    "RepeatTraversal",
    "Query",
    "HoleQuery",
    "HybridQuery",
]


# type alias
Vocabularies = Dict[Text, List[Text]]


class AstNode:
    """The base class for all AST nodes."""

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self}>"

    def __eq__(self, value):
        return self.id_tuple() == value.id_tuple()

    def __hash__(self):
        return hash(self.id_tuple())

    def id_tuple(self):
        return (type(self),)

    def is_hole(self) -> bool:
        """Returns true if the node is a hole."""
        # most nodes are not holes,
        # so only the Hole* nodes need to override this
        return False

    def has_holes(self) -> bool:
        """Returns true if the pattern has one or more holes."""
        # most nodes need to override this to handle their children,
        # so the default implementation is intended for Hole* nodes
        return self.is_hole()

    def is_valid(self) -> bool:
        """Returns true if the pattern is valid, i.e., has no holes."""
        return not self.has_holes()

    def tokens(self) -> List[Text]:
        """Returns the pattern as a list of tokens."""
        # default implementation is intended for nodes that have no children
        return [Text(self)]

    def num_matcher_holes(self) -> int:
        """Returns the number of matcher holes in this pattern."""
        return 0

    def num_constraint_holes(self) -> int:
        """Returns the number of constraint holes in this pattern."""
        return 0

    def num_surface_holes(self) -> int:
        """Returns the number of surface holes in this pattern."""
        return 0

    def num_traversal_holes(self) -> int:
        """Returns the number of traversal holes in this pattern."""
        return 0

    def num_query_holes(self) -> int:
        """Returns the number of traversal holes in this pattern."""
        return 0

    def num_holes(self) -> int:
        """Returns the number of holes in this pattern."""
        return (
            self.num_matcher_holes()
            + self.num_constraint_holes()
            + self.num_surface_holes()
            + self.num_traversal_holes()
            + self.num_query_holes()
        )

    def expand_leftmost_hole(
        self, vocabularies: Vocabularies, **kwargs
    ) -> List[AstNode]:
        """
        If the pattern has holes then it returns the patterns obtained
        by expanding the leftmost hole.  If there are no holes then it
        returns an empty list.
        """
        # default implementation is suitable for Matchers only
        return []

    def preorder_traversal(self) -> List[AstNode]:
        """Returns a list with all the nodes of the tree in preorder."""
        # default implementation is for nodes that have no children
        return [self]

    def permutations(self) -> List[AstNode]:
        """Returns all trees that are equivalent to this AstNode."""
        return [self]


# type alias
Types = Type[Union[AstNode, Tuple[AstNode]]]


def is_identifier(s: Text) -> bool:
    """returns true if the provided string is a valid identifier"""
    return config.IDENT_RE.match(s) is not None


def maybe_parens(node: AstNode, types: Types) -> str:
    """Converts node to string. Surrounds by parenthesis
    if node is subclass of provided types."""
    return f"({node})" if isinstance(node, types) else str(node)


def maybe_parens_tokens(node: AstNode, types: Types) -> List[Text]:
    """Converts node to list of tokens. Surrounds by parenthesis
    if node is subclass of provided types."""
    return ["(", *node.tokens(), ")"] if isinstance(node, types) else node.tokens()


def make_quantifier(min: int, max: Optional[int]) -> str:
    """Gets the desired minimum and maximum repetitions
    and returns the appropriate quantifier."""
    return "".join(make_quantifier_tokens(min, max))


def make_quantifier_tokens(min: int, max: Optional[int]) -> List[Text]:
    """Gets the desired minimum and maximum repetitions
    and returns the sequence of tokens corresponding
    to the appropriate quantifier."""
    if min == max:
        return ["{", str(min), "}"]
    if max == None:
        if min == 0:
            return ["*"]
        elif min == 1:
            return ["+"]
        else:
            return ["{", str(min), ",", "}"]
    if min == 0:
        if max == 1:
            return ["?"]
        else:
            return ["{", ",", str(max), "}"]
    return ["{", str(min), ",", str(max), "}"]


def all_binary_trees(nodes: List[AstNode], cls: Type) -> List[AstNode]:
    """Returns all the binary trees of type `cls` that can be constructed
    with the given nodes."""
    if len(nodes) == 1:
        return nodes
    trees = []
    for i in range(1, len(nodes)):
        for l in all_binary_trees(nodes[:i], cls):
            for r in all_binary_trees(nodes[i:], cls):
                trees.append(cls(l, r))
    return trees


def get_clauses(node, cls=None):
    """Flattens and returns the clauses of the given node."""
    clauses = []
    if cls is None:
        cls = type(node)
    if isinstance(node.lhs, cls):
        clauses += get_clauses(node.lhs, cls)
    else:
        clauses.append(node.lhs)
    if isinstance(node.rhs, cls):
        clauses += get_clauses(node.rhs, cls)
    else:
        clauses.append(node.rhs)
    return clauses


def get_all_trees(node: AstNode) -> List[AstNode]:
    """Returns all equivalent trees to node."""
    results = []
    cls = type(node)
    perms_per_clause = [c.permutations() for c in get_clauses(node)]
    for clauses in itertools.product(*perms_per_clause):
        results += all_binary_trees(clauses, cls)
    return results


####################
# string matchers
####################


class Matcher(AstNode):
    """The base class for all string matchers."""


class HoleMatcher(Matcher):
    def __str__(self):
        return config.SURFACE_HOLE_GLYPH

    def is_hole(self):
        return True

    def num_matcher_holes(self):
        return 1


class ExactMatcher(Matcher):
    def __init__(self, s: Text):
        self.string = s

    def __str__(self):
        if is_identifier(self.string):
            # don't surround identifiers with quotes
            return self.string
        else:
            return json.dumps(self.string)

    def id_tuple(self):
        return super().id_tuple() + (self.string,)


####################
# token constraints
####################


class Constraint(AstNode):
    """The base class for all token constraints."""


class HoleConstraint(Constraint):
    def __str__(self):
        return config.SURFACE_HOLE_GLYPH

    def is_hole(self):
        return True

    def num_constraint_holes(self):
        return 1

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        return [
            FieldConstraint(HoleMatcher(), HoleMatcher()),
            NotConstraint(HoleConstraint()),
            AndConstraint(HoleConstraint(), HoleConstraint()),
            OrConstraint(HoleConstraint(), HoleConstraint()),
        ]


class FieldConstraint(Constraint):
    def __init__(self, name: Matcher, value: Matcher):
        self.name = name
        self.value = value

    def __str__(self):
        return f"{self.name}={self.value}"

    def id_tuple(self):
        return super().id_tuple() + (self.name, self.value)

    def has_holes(self):
        return self.name.has_holes() or self.value.has_holes()

    def tokens(self):
        return self.name.tokens() + ["="] + self.value.tokens()

    def num_matcher_holes(self):
        return self.name.num_matcher_holes() + self.value.num_matcher_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.name.is_hole():
            return [
                FieldConstraint(ExactMatcher(name), self.value)
                for name in vocabularies
                if name not in config.EXCLUDE_FIELDS
            ]
        elif self.value.is_hole():
            return [
                FieldConstraint(self.name, ExactMatcher(value))
                for value in vocabularies[self.name.string]
            ]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.name.preorder_traversal()
            + self.value.preorder_traversal()
        )


class NotConstraint(Constraint):
    def __init__(self, c: Constraint):
        self.constraint = c

    def __str__(self):
        c = maybe_parens(self.constraint, (AndConstraint, OrConstraint))
        return f"!{c}"

    def id_tuple(self):
        return super().id_tuple() + (self.constraint,)

    def has_holes(self):
        return self.constraint.has_holes()

    def tokens(self):
        return ["!"] + maybe_parens_tokens(
            self.constraint, (AndConstraint, OrConstraint)
        )

    def num_matcher_holes(self):
        return self.constraint.num_matcher_holes()

    def num_constraint_holes(self):
        return self.constraint.num_constraint_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        # get the next nodes for the nested constraint
        nodes = self.constraint.expand_leftmost_hole(vocabularies, **kwargs)
        # avoid nesting negations
        return [NotConstraint(n) for n in nodes if not isinstance(n, NotConstraint)]

    def preorder_traversal(self):
        return super().preorder_traversal() + self.constraint.preorder_traversal()

    def permutations(self):
        return [NotConstraint(p) for p in self.constraint.permutations()]


class AndConstraint(Constraint):
    def __init__(self, lhs: Constraint, rhs: Constraint):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} & {self.rhs}"

    def id_tuple(self):
        return super().id_tuple() + (self.lhs, self.rhs)

    def has_holes(self):
        return self.lhs.has_holes() or self.rhs.has_holes()

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.lhs, OrConstraint)
        tokens.append("&")
        tokens += maybe_parens_tokens(self.rhs, OrConstraint)
        return tokens

    def num_matcher_holes(self):
        return self.lhs.num_matcher_holes() + self.rhs.num_matcher_holes()

    def num_constraint_holes(self):
        return self.lhs.num_constraint_holes() + self.rhs.num_constraint_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [AndConstraint(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [AndConstraint(self.lhs, n) for n in nodes]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.lhs.preorder_traversal()
            + self.rhs.preorder_traversal()
        )

    def permutations(self):
        return get_all_trees(self)


class OrConstraint(Constraint):
    def __init__(self, lhs: Constraint, rhs: Constraint):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} | {self.rhs}"

    def id_tuple(self):
        return super().id_tuple() + (self.lhs, self.rhs)

    def has_holes(self):
        return self.lhs.has_holes() or self.rhs.has_holes()

    def tokens(self):
        return [*self.lhs.tokens(), "|", *self.rhs.tokens()]

    def num_matcher_holes(self):
        return self.lhs.num_matcher_holes() + self.rhs.num_matcher_holes()

    def num_constraint_holes(self):
        return self.lhs.num_constraint_holes() + self.rhs.num_constraint_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [OrConstraint(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [OrConstraint(self.lhs, n) for n in nodes]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.lhs.preorder_traversal()
            + self.rhs.preorder_traversal()
        )

    def permutations(self):
        return get_all_trees(self)


####################
# surface patterns
####################


class Surface(AstNode):
    """The base class for all surface patterns."""


class HoleSurface(Surface):
    def __str__(self):
        return config.SURFACE_HOLE_GLYPH

    def is_hole(self):
        return True

    def num_surface_holes(self):
        return 1

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        candidates = [
            TokenSurface(HoleConstraint()),
        ]
        if kwargs.get("allow_surface_wildcards", True):
            candidates.append(WildcardSurface())
        if kwargs.get("allow_surface_mentions", True):
            candidates.append(MentionSurface(HoleMatcher()))
        if kwargs.get("allow_surface_alternations", True):
            candidates.append(OrSurface(HoleSurface(), HoleSurface()))
        if kwargs.get("allow_surface_concatenations", True):
            candidates.append(ConcatSurface(HoleSurface(), HoleSurface()))
        if kwargs.get("allow_surface_repetitions", True):
            candidates += [
                RepeatSurface(HoleSurface(), 0, 1),
                RepeatSurface(HoleSurface(), 0, None),
                RepeatSurface(HoleSurface(), 1, None),
            ]
        return candidates


class WildcardSurface(Surface):
    def __str__(self):
        return "[]"

    def tokens(self):
        return ["[", "]"]


class TokenSurface(Surface):
    def __init__(self, c: Constraint):
        self.constraint = c

    def __str__(self):
        return f"[{self.constraint}]"

    def id_tuple(self):
        return super().id_tuple() + (self.constraint,)

    def has_holes(self):
        return self.constraint.has_holes()

    def tokens(self):
        return ["[", *self.constraint.tokens(), "]"]

    def num_matcher_holes(self):
        return self.constraint.num_matcher_holes()

    def num_constraint_holes(self):
        return self.constraint.num_constraint_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        nodes = self.constraint.expand_leftmost_hole(vocabularies, **kwargs)
        return [TokenSurface(n) for n in nodes]

    def preorder_traversal(self):
        return super().preorder_traversal() + self.constraint.preorder_traversal()

    def permutations(self):
        return [TokenSurface(p) for p in self.constraint.permutations()]


class MentionSurface(Surface):
    def __init__(self, label: Matcher):
        self.label = label

    def __str__(self):
        return f"@{self.label}"

    def id_tuple(self):
        return super().id_tuple() + (self.label,)

    def has_holes(self):
        return self.label.has_holes()

    def tokens(self):
        return ["@"] + self.label.tokens()

    def num_matcher_holes(self):
        return self.label.num_matcher_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        entities = vocabularies.get(config.ENTITY_FIELD, [])
        return [MentionSurface(ExactMatcher(e)) for e in entities]

    def preorder_traversal(self):
        return super().preorder_traversal() + self.label.preorder_traversal()


class ConcatSurface(Surface):
    def __init__(self, lhs: Surface, rhs: Surface):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        lhs = maybe_parens(self.lhs, OrSurface)
        rhs = maybe_parens(self.rhs, OrSurface)
        return f"{lhs} {rhs}"

    def id_tuple(self):
        return super().id_tuple() + (self.lhs, self.rhs)

    def has_holes(self):
        return self.lhs.has_holes() or self.rhs.has_holes()

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.lhs, OrSurface)
        tokens += maybe_parens_tokens(self.rhs, OrSurface)
        return tokens

    def num_matcher_holes(self):
        return self.lhs.num_matcher_holes() + self.rhs.num_matcher_holes()

    def num_constraint_holes(self):
        return self.lhs.num_constraint_holes() + self.rhs.num_constraint_holes()

    def num_surface_holes(self):
        return self.lhs.num_surface_holes() + self.rhs.num_surface_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [ConcatSurface(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [ConcatSurface(self.lhs, n) for n in nodes]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.lhs.preorder_traversal()
            + self.rhs.preorder_traversal()
        )

    def permutations(self):
        return get_all_trees(self)


class OrSurface(Surface):
    def __init__(self, lhs: Surface, rhs: Surface):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} | {self.rhs}"

    def id_tuple(self):
        return super().id_tuple() + (self.lhs, self.rhs)

    def has_holes(self):
        return self.lhs.has_holes() or self.rhs.has_holes()

    def tokens(self):
        return [*self.lhs.tokens(), "|", *self.rhs.tokens()]

    def num_matcher_holes(self):
        return self.lhs.num_matcher_holes() + self.rhs.num_matcher_holes()

    def num_constraint_holes(self):
        return self.lhs.num_constraint_holes() + self.rhs.num_constraint_holes()

    def num_surface_holes(self):
        return self.lhs.num_surface_holes() + self.rhs.num_surface_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [OrSurface(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [OrSurface(self.lhs, n) for n in nodes]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.lhs.preorder_traversal()
            + self.rhs.preorder_traversal()
        )

    def permutations(self):
        return get_all_trees(self)


class RepeatSurface(Surface):
    def __init__(self, surf: Surface, min: int, max: Optional[int]):
        self.surf = surf
        self.min = min
        self.max = max

    def __str__(self):
        surf = maybe_parens(self.surf, (ConcatSurface, OrSurface))
        quant = make_quantifier(self.min, self.max)
        return f"{surf}{quant}"

    def id_tuple(self):
        return super().id_tuple() + (self.surf, self.min, self.max)

    def has_holes(self):
        return self.surf.has_holes()

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.surf, (ConcatSurface, OrSurface))
        tokens += make_quantifier_tokens(self.min, self.max)
        return tokens

    def num_matcher_holes(self):
        return self.surf.num_matcher_holes()

    def num_constraint_holes(self):
        return self.surf.num_constraint_holes()

    def num_surface_holes(self):
        return self.surf.num_surface_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        nodes = self.surf.expand_leftmost_hole(vocabularies, **kwargs)
        # avoid nesting repetitions
        nodes = [n for n in nodes if not isinstance(n, RepeatSurface)]
        return [RepeatSurface(n, self.min, self.max) for n in nodes]

    def preorder_traversal(self):
        return super().preorder_traversal() + self.surf.preorder_traversal()

    def permutations(self):
        return [RepeatSurface(p, self.min, self.max) for p in self.surf.permutations()]


####################
# traversal patterns
####################


class Traversal(AstNode):
    """The base class for all graph traversals."""


class HoleTraversal(Traversal):
    def __str__(self):
        return config.TRAVERSAL_HOLE_GLYPH

    def is_hole(self):
        return True

    def num_traversal_holes(self):
        return 1

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        candidates = [
            IncomingLabelTraversal(HoleMatcher()),
            OutgoingLabelTraversal(HoleMatcher()),
        ]
        if kwargs.get("allow_traversal_wildcards", True):
            candidates += [
                IncomingWildcardTraversal(),
                OutgoingWildcardTraversal(),
            ]
        if kwargs.get("allow_traversal_alternations", True):
            candidates.append(OrTraversal(HoleTraversal(), HoleTraversal()))
        if kwargs.get("allow_traversal_concatenations", True):
            candidates.append(ConcatTraversal(HoleTraversal(), HoleTraversal()))
        if kwargs.get("allow_traversal_repetitions", True):
            candidates += [
                RepeatTraversal(HoleTraversal(), 0, 1),
                RepeatTraversal(HoleTraversal(), 0, None),
                RepeatTraversal(HoleTraversal(), 1, None),
            ]
        return candidates


class IncomingWildcardTraversal(Traversal):
    def __str__(self):
        return "<<"

    def tokens(self):
        return ["<<"]


class OutgoingWildcardTraversal(Traversal):
    def __str__(self):
        return ">>"

    def tokens(self):
        return [">>"]


class IncomingLabelTraversal(Traversal):
    def __init__(self, label: Matcher):
        self.label = label

    def __str__(self):
        return f"<{self.label}"

    def id_tuple(self):
        return super().id_tuple() + (self.label,)

    def has_holes(self):
        return self.label.has_holes()

    def tokens(self):
        return ["<"] + self.label.tokens()

    def num_matcher_holes(self):
        return self.label.num_matcher_holes()

    def num_traversal_holes(self):
        return self.label.num_traversal_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.label.is_hole():
            return [
                IncomingLabelTraversal(ExactMatcher(v))
                for v in vocabularies.get(config.SYNTAX_FIELD, [])
            ]
        else:
            return []

    def preorder_traversal(self):
        return super().preorder_traversal() + self.label.preorder_traversal()


class OutgoingLabelTraversal(Traversal):
    def __init__(self, label: Matcher):
        self.label = label

    def __str__(self):
        return f">{self.label}"

    def id_tuple(self):
        return super().id_tuple() + (self.label,)

    def has_holes(self):
        return self.label.has_holes()

    def tokens(self):
        return [">"] + self.label.tokens()

    def num_matcher_holes(self):
        return self.label.num_matcher_holes()

    def num_traversal_holes(self):
        return self.label.num_traversal_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.label.is_hole():
            return [
                OutgoingLabelTraversal(ExactMatcher(v))
                for v in vocabularies.get(config.SYNTAX_FIELD, [])
            ]
        else:
            return []

    def preorder_traversal(self):
        return super().preorder_traversal() + self.label.preorder_traversal()


class ConcatTraversal(Traversal):
    def __init__(self, lhs: Traversal, rhs: Traversal):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        lhs = maybe_parens(self.lhs, OrTraversal)
        rhs = maybe_parens(self.rhs, OrTraversal)
        return f"{lhs} {rhs}"

    def id_tuple(self):
        return super().id_tuple() + (self.lhs, self.rhs)

    def has_holes(self):
        return self.lhs.has_holes() or self.rhs.has_holes()

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.lhs, OrTraversal)
        tokens += maybe_parens_tokens(self.rhs, OrTraversal)
        return tokens

    def num_matcher_holes(self):
        return self.lhs.num_matcher_holes() + self.rhs.num_matcher_holes()

    def num_traversal_holes(self):
        return self.lhs.num_traversal_holes() + self.rhs.num_traversal_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [ConcatTraversal(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [ConcatTraversal(self.lhs, n) for n in nodes]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.lhs.preorder_traversal()
            + self.rhs.preorder_traversal()
        )

    def permutations(self):
        return get_all_trees(self)


class OrTraversal(Traversal):
    def __init__(self, lhs: Traversal, rhs: Traversal):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} | {self.rhs}"

    def id_tuple(self):
        return super().id_tuple() + (self.lhs, self.rhs)

    def has_holes(self):
        return self.lhs.has_holes() or self.rhs.has_holes()

    def tokens(self):
        return self.lhs.tokens() + ["|"] + self.rhs.tokens()

    def num_matcher_holes(self):
        return self.lhs.num_matcher_holes() + self.rhs.num_matcher_holes()

    def num_traversal_holes(self):
        return self.lhs.num_traversal_holes() + self.rhs.num_traversal_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [OrTraversal(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [OrTraversal(self.lhs, n) for n in nodes]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.lhs.preorder_traversal()
            + self.rhs.preorder_traversal()
        )

    def permutations(self):
        return get_all_trees(self)


class RepeatTraversal(Traversal):
    def __init__(self, traversal: Traversal, min: int, max: Optional[int]):
        self.traversal = traversal
        self.min = min
        self.max = max

    def __str__(self):
        traversal = maybe_parens(self.traversal, (ConcatTraversal, OrTraversal))
        quant = make_quantifier(self.min, self.max)
        return f"{traversal}{quant}"

    def id_tuple(self):
        return super().id_tuple() + (self.traversal, self.min, self.max)

    def has_holes(self):
        return self.traversal.has_holes()

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.traversal, (ConcatTraversal, OrTraversal))
        tokens += make_quantifier_tokens(self.min, self.max)
        return tokens

    def num_matcher_holes(self):
        return self.traversal.num_matcher_holes()

    def num_traversal_holes(self):
        return self.traversal.num_traversal_holes()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        nodes = self.traversal.expand_leftmost_hole(vocabularies, **kwargs)
        nodes = [n for n in nodes if not isinstance(n, RepeatTraversal)]
        return [RepeatTraversal(n, self.min, self.max) for n in nodes]

    def preorder_traversal(self):
        return super().preorder_traversal() + self.traversal.preorder_traversal()

    def permutations(self):
        return [
            RepeatTraversal(p, self.min, self.max)
            for p in self.traversal.permutations()
        ]


####################
# query
####################


class Query(AstNode):
    """The base class for hybrid queries."""


class HoleQuery(Query):
    def __str__(self):
        return config.QUERY_HOLE_GLYPH

    def is_hole(self):
        return True

    def num_query_holes(self):
        return 1

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        return [
            HoleSurface(),
            HybridQuery(HoleSurface(), HoleTraversal(), HoleQuery()),
        ]


class HybridQuery(Query):
    def __init__(self, src: Surface, traversal: Traversal, dst: AstNode):
        self.src = src
        self.dst = dst
        self.traversal = traversal

    def __str__(self):
        src = maybe_parens(self.src, OrSurface)
        dst = maybe_parens(self.dst, OrSurface)
        traversal = maybe_parens(self.traversal, OrTraversal)
        return f"{src} {traversal} {dst}"

    def id_tuple(self):
        return super().id_tuple() + (self.src, self.traversal, self.dst)

    def has_holes(self):
        return (
            self.src.has_holes() or self.traversal.has_holes() or self.dst.has_holes()
        )

    def tokens(self):
        src = maybe_parens_tokens(self.src, OrSurface)
        dst = maybe_parens_tokens(self.dst, OrSurface)
        traversal = maybe_parens_tokens(self.traversal, OrTraversal)
        return src + traversal + dst

    def num_matcher_holes(self):
        return (
            self.src.num_matcher_holes()
            + self.traversal.num_matcher_holes()
            + self.dst.num_matcher_holes()
        )

    def num_constraint_holes(self):
        return (
            self.src.num_constraint_holes()
            + self.traversal.num_constraint_holes()
            + self.dst.num_constraint_holes()
        )

    def num_surface_holes(self):
        return (
            self.src.num_surface_holes()
            + self.traversal.num_surface_holes()
            + self.dst.num_surface_holes()
        )

    def num_traversal_holes(self):
        return (
            self.src.num_traversal_holes()
            + self.traversal.num_traversal_holes()
            + self.dst.num_traversal_holes()
        )

    def num_query_holes(self):
        return (
            self.src.num_query_holes()
            + self.traversal.num_query_holes()
            + self.dst.num_query_holes()
        )

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.src.has_holes():
            nodes = self.src.expand_leftmost_hole(vocabularies, **kwargs)
            return [HybridQuery(n, self.traversal, self.dst) for n in nodes]
        elif self.traversal.has_holes():
            nodes = self.traversal.expand_leftmost_hole(vocabularies, **kwargs)
            return [HybridQuery(self.src, n, self.dst) for n in nodes]
        elif self.dst.has_holes():
            nodes = self.dst.expand_leftmost_hole(vocabularies, **kwargs)
            return [HybridQuery(self.src, self.traversal, n) for n in nodes]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.src.preorder_traversal()
            + self.traversal.preorder_traversal()
            + self.dst.preorder_traversal()
        )

    def permutations(self):
        return [
            HybridQuery(src, traversal, dst)
            for src in self.src.permutations()
            for traversal in self.traversal.permutations()
            for dst in self.dst.permutations()
        ]
