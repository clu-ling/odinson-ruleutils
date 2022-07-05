from __future__ import annotations
import json
import math
import itertools
from typing import List, Text, Optional, Tuple, Type, Union
from odinson.ruleutils import config

__all__ = [
    "Vocabularies",
    "AstNode",
    "Matcher",
    "HoleMatcher",
    "ExactMatcher",
    "Constraint",
    "HoleConstraint",
    "WildcardConstraint",
    "FieldConstraint",
    "NotConstraint",
    "AndConstraint",
    "OrConstraint",
    "Surface",
    "HoleSurface",
    "TokenSurface",
    "MentionSurface",
    "WildcardSurface",
    "ConcatSurface",
    "OrSurface",
    "RepeatSurface",
    "Traversal",
    "HoleTraversal",
    "IncomingWildcardTraversal",
    "OutgoingWildcardTraversal",
    "RepeatTraversal",
    "Query",
    "HoleQuery",
    "HybridQuery",
    "IncomingLabelTraversal",
    "OutgoingLabelTraversal",
    "ConcatTraversal",
    "OrTraversal",
    "RepeatTraversal",
    "Query",
    "HoleQuery",
    "HybridQuery",
]


# type alias
Vocabularies = config.Vocabularies


OPERATORS_TO_EXCLUDE = {"]", ")", "}"}


OPERATORS = {
    "[",
    "(",
    "{",
    "?",
    "*",
    "+",
    "=",
    "!",
    "&",
    "|",
    ",",
    "@",
    "<",
    ">",
    ">>",
    "<<",
}


class CognitiveWeight:
    FIELD_CONSTRAINT = 1
    NOT_CONSTRAINT = 2
    AND_CONSTRAINT = 3
    OR_CONSTRAINT = 4
    WILDCARD_SURFACE = 1
    TOKEN_SURFACE = 1
    MENTION_SURFACE = 1
    CONCAT_SURFACE = 3
    OR_SURFACE = 4
    REPEAT_SURFACE = 5
    INCOMING_WILDCARD_TRAVERSAL = 1
    OUTGOING_WILDCARD_TRAVERSAL = 1
    INCOMING_LABEL_TRAVERSAL = 1
    OUTGOING_LABEL_TRAVERSAL = 1
    CONCAT_TRAVERSAL = 3
    OR_TRAVERSAL = 4
    REPEAT_TRAVERSAL = 5
    HYBRID_QUERY = 2


class AstNode:
    """The base class for all AST nodes."""

    def __repr__(self):
        return f"<{self.__class__.__name__}: {str(self)!r}>"

    def __eq__(self, value):
        return self.id_tuple() == value.id_tuple()

    def __hash__(self):
        return hash(self.id_tuple())

    def children(self):
        return []

    def id_tuple(self):
        return (type(self), *self.children())

    def is_hole(self) -> bool:
        """Returns true if the node is a hole."""
        # most nodes are not holes,
        # so only the Hole* nodes need to override this
        return False

    def has_holes(self) -> bool:
        """Returns true if the pattern has one or more holes."""
        # most nodes need to override this to handle their children,
        # so the default implementation is intended for Hole* nodes
        return self.is_hole() or any(c.has_holes() for c in self.children())

    def is_valid(self) -> bool:
        """Returns true if the pattern is valid, i.e., has no holes."""
        return not self.has_holes()

    def tokens(self) -> List[Text]:
        """Returns the pattern as a list of tokens."""
        # default implementation is intended for nodes that have no children
        return [Text(self)]

    def num_matcher_holes(self) -> int:
        """Returns the number of matcher holes in this pattern."""
        return sum(c.num_matcher_holes() for c in self.children())

    def num_constraint_holes(self) -> int:
        """Returns the number of constraint holes in this pattern."""
        return sum(c.num_constraint_holes() for c in self.children())

    def num_surface_holes(self) -> int:
        """Returns the number of surface holes in this pattern."""
        return sum(c.num_surface_holes() for c in self.children())

    def num_traversal_holes(self) -> int:
        """Returns the number of traversal holes in this pattern."""
        return sum(c.num_traversal_holes() for c in self.children())

    def num_query_holes(self) -> int:
        """Returns the number of traversal holes in this pattern."""
        return sum(c.num_query_holes() for c in self.children())

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
        nodes = [self]
        for child in self.children():
            nodes += child.preorder_traversal()
        return nodes

    def permutations(self) -> List[AstNode]:
        """Returns all trees that are equivalent to this AstNode."""
        return [self]

    def over_approximation(self) -> Optional[AstNode]:
        """Returns a rule with a language that contains all the languages
        of the current node's descendents."""
        return self

    def under_approximation(self) -> Optional[AstNode]:
        """Returns a rule with a language that is subsumed by the languages
        of all the descendents of the current node"""
        return self

    def redundancy_patterns(self) -> List[AstNode]:
        return [n.over_approximation() for n in set(self.unroll().split())]

    def unroll(self) -> AstNode:
        """unroll repetitions"""
        return self

    def split(self) -> List[AstNode]:
        """decompose rule"""
        return [self]

    _COGNITIVE_WEIGHT = 0

    def cognitive_weight(self) -> int:
        return self._COGNITIVE_WEIGHT + sum(
            c.cognitive_weight() for c in self.children()
        )

    def operators(self) -> list[str]:
        return [t for t in self.tokens() if t in OPERATORS]

    def operands(self) -> list[str]:
        return [
            t
            for t in self.tokens()
            if t not in OPERATORS and t not in OPERATORS_TO_EXCLUDE
        ]

    def num_operators(self) -> int:
        return len(self.operators())

    def num_distinct_operators(self) -> int:
        return len(set(self.operators()))

    def num_operands(self) -> int:
        return len(self.operands())

    def num_distinct_operands(self) -> int:
        return len(set(self.operands()))

    def implementation_length(self) -> int:
        return self.num_operators() + self.num_operands()

    def vocabulary_length(self) -> int:
        return self.num_distinct_operators() + self.num_distinct_operands()

    def program_length(self) -> float:
        operators = self.num_distinct_operators()
        operands = self.num_distinct_operands()
        return operators * math.log(operators, 2) + operands * math.log(operands, 2)

    def program_volume(self) -> float:
        return self.implementation_length() * math.log(self.vocabulary_length(), 2)

    def potential_volume(self) -> float:
        # NOTE this may not be correct for our language
        x = 2 + self.num_distinct_operands()
        return x * math.log(x, 2)

    def program_level(self) -> float:
        return self.potential_volume() / self.program_volume()

    def effort(self) -> float:
        return self.program_volume() / self.program_level()

    def number_incomings(self) -> int:
        return len([t for t in self.tokens() if t.startswith("<")])

    def number_outgoings(self) -> int:
        return len([t for t in self.tokens() if t.startswith(">")])

    def proportion_incoming(self) -> float:
        n_in = self.number_incomings()
        n_out = self.number_outgoings()
        return n_in / (n_in + n_out)

    def num_quantifiers(self):
        return sum(c.num_quantifiers() for c in self.children())

    def num_nodes(self) -> int:
        return 1 + sum(c.num_nodes() for c in self.children())

    def num_leaves(self) -> int:
        children = self.children()
        return 1 if not children else sum(c.num_leaves() for c in children)

    def tree_height(self, func):
        height = 0
        children = self.children()
        if children:
            height = func(c.tree_height(func) for c in children)
        return height + 1

    def max_tree_height(self) -> int:
        return self.tree_height(max)

    def min_tree_height(self) -> int:
        return self.tree_height(min)


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

    def over_approximation(self):
        return WildcardMatcher()

    def under_approximation(self):
        return None


class WildcardMatcher(Matcher):
    def __str__(self):
        # this should never be rendered
        return "???"


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


class WildcardConstraint(Constraint):
    def __str__(self):
        # this should never be rendered
        return "???"


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

    def over_approximation(self):
        return WildcardConstraint()

    def under_approximation(self):
        return None


class FieldConstraint(Constraint):
    _COGNITIVE_WEIGHT = CognitiveWeight.FIELD_CONSTRAINT

    def __init__(self, name: Matcher, value: Matcher):
        self.name = name
        self.value = value

    def __str__(self):
        return f"{self.name}={self.value}"

    def children(self):
        return [self.name, self.value]

    def tokens(self):
        return self.name.tokens() + ["="] + self.value.tokens()

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

    def over_approximation(self):
        name = self.name.over_approximation()
        if name is None:
            return None
        if isinstance(name, WildcardMatcher):
            return WildcardConstraint()
        value = self.value.over_approximation()
        if value is None:
            return None
        if isinstance(value, WildcardMatcher):
            return WildcardConstraint()
        return FieldConstraint(name, value)

    def under_approximation(self):
        name = self.name.under_approximation()
        if name is None:
            return None
        if isinstance(name, WildcardMatcher):
            return WildcardConstraint()
        value = self.value.under_approximation()
        if value is None:
            return None
        if isinstance(value, WildcardMatcher):
            return WildcardConstraint()
        return FieldConstraint(name, value)


class NotConstraint(Constraint):
    _COGNITIVE_WEIGHT = CognitiveWeight.NOT_CONSTRAINT

    def __init__(self, c: Constraint):
        self.constraint = c

    def __str__(self):
        c = maybe_parens(self.constraint, (AndConstraint, OrConstraint))
        return f"!{c}"

    def children(self):
        return [self.constraint]

    def tokens(self):
        return ["!"] + maybe_parens_tokens(
            self.constraint, (AndConstraint, OrConstraint)
        )

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        # get the next nodes for the nested constraint
        nodes = self.constraint.expand_leftmost_hole(vocabularies, **kwargs)
        # avoid nesting negations
        return [NotConstraint(n) for n in nodes if not isinstance(n, NotConstraint)]

    def permutations(self):
        return [NotConstraint(p) for p in self.constraint.permutations()]

    def over_approximation(self):
        constraint = self.constraint.over_approximation()
        if constraint is None:
            return WildcardConstraint()
        if isinstance(constraint, WildcardConstraint):
            return None
        return NotConstraint(constraint)

    def under_approximation(self):
        constraint = self.constraint.under_approximation()
        if constraint is None:
            return WildcardConstraint()
        if isinstance(constraint, WildcardConstraint):
            return None
        return NotConstraint(constraint)


class AndConstraint(Constraint):
    _COGNITIVE_WEIGHT = CognitiveWeight.AND_CONSTRAINT

    def __init__(self, lhs: Constraint, rhs: Constraint):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} & {self.rhs}"

    def children(self):
        return [self.lhs, self.rhs]

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.lhs, OrConstraint)
        tokens.append("&")
        tokens += maybe_parens_tokens(self.rhs, OrConstraint)
        return tokens

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [AndConstraint(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [AndConstraint(self.lhs, n) for n in nodes]
        else:
            return []

    def permutations(self):
        return get_all_trees(self)

    def over_approximation(self):
        lhs = self.lhs.over_approximation()
        rhs = self.rhs.over_approximation()
        if lhs is None or rhs is None:
            return None
        if isinstance(lhs, WildcardConstraint):
            return rhs
        if isinstance(rhs, WildcardConstraint):
            return lhs
        return AndConstraint(lhs, rhs)

    def under_approximation(self):
        lhs = self.lhs.under_approximation()
        rhs = self.rhs.under_approximation()
        if lhs is None or rhs is None:
            return None
        if isinstance(lhs, WildcardConstraint):
            return rhs
        if isinstance(rhs, WildcardConstraint):
            return lhs
        return AndConstraint(lhs, rhs)


class OrConstraint(Constraint):
    _COGNITIVE_WEIGHT = CognitiveWeight.OR_CONSTRAINT

    def __init__(self, lhs: Constraint, rhs: Constraint):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} | {self.rhs}"

    def children(self):
        return [self.lhs, self.rhs]

    def tokens(self):
        return [*self.lhs.tokens(), "|", *self.rhs.tokens()]

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [OrConstraint(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [OrConstraint(self.lhs, n) for n in nodes]
        else:
            return []

    def permutations(self):
        return get_all_trees(self)

    def over_approximation(self):
        lhs = self.lhs.over_approximation()
        rhs = self.rhs.over_approximation()
        if lhs is None:
            return rhs
        if rhs is None:
            return lhs
        if isinstance(lhs, WildcardConstraint) or isinstance(rhs, WildcardConstraint):
            return WildcardConstraint()
        return OrConstraint(lhs, rhs)

    def under_approximation(self):
        lhs = self.lhs.under_approximation()
        rhs = self.rhs.under_approximation()
        if lhs is None:
            return rhs
        if rhs is None:
            return lhs
        if isinstance(lhs, WildcardConstraint) or isinstance(rhs, WildcardConstraint):
            return WildcardConstraint()
        return OrConstraint(lhs, rhs)

    def split(self):
        return self.lhs.split() + self.rhs.split()


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
        if (
            kwargs.get("allow_surface_mentions", True)
            and config.ENTITY_FIELD in vocabularies
        ):
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

    def over_approximation(self):
        return RepeatSurface(WildcardSurface(), 0, None)

    def under_approximation(self):
        return None


class WildcardSurface(Surface):
    _COGNITIVE_WEIGHT = CognitiveWeight.WILDCARD_SURFACE

    def __str__(self):
        return "[]"

    def tokens(self):
        return ["[", "]"]


class TokenSurface(Surface):
    _COGNITIVE_WEIGHT = CognitiveWeight.TOKEN_SURFACE

    def __init__(self, c: Constraint):
        self.constraint = c

    def __str__(self):
        return f"[{self.constraint}]"

    def children(self):
        return [self.constraint]

    def tokens(self):
        return ["[", *self.constraint.tokens(), "]"]

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        nodes = self.constraint.expand_leftmost_hole(vocabularies, **kwargs)
        return [TokenSurface(n) for n in nodes]

    def permutations(self):
        return [TokenSurface(p) for p in self.constraint.permutations()]

    def over_approximation(self):
        constraint = self.constraint.over_approximation()
        if constraint is None:
            return None
        if isinstance(constraint, WildcardConstraint):
            return WildcardSurface()
        return TokenSurface(constraint)

    def under_approximation(self):
        constraint = self.constraint.under_approximation()
        if constraint is None:
            return None
        if isinstance(constraint, WildcardConstraint):
            return WildcardSurface()
        return TokenSurface(constraint)

    def unroll(self):
        return TokenSurface(self.constraint.unroll())

    def split(self):
        return [TokenSurface(c) for c in self.constraint.split()]


class MentionSurface(Surface):
    _COGNITIVE_WEIGHT = CognitiveWeight.MENTION_SURFACE

    def __init__(self, label: Matcher):
        self.label = label

    def __str__(self):
        return f"@{self.label}"

    def children(self):
        return [self.label]

    def tokens(self):
        return ["@"] + self.label.tokens()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        entities = vocabularies.get(config.ENTITY_FIELD, [])
        return [MentionSurface(ExactMatcher(e)) for e in entities]


class ConcatSurface(Surface):
    _COGNITIVE_WEIGHT = CognitiveWeight.CONCAT_SURFACE

    def __init__(self, lhs: Surface, rhs: Surface):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        lhs = maybe_parens(self.lhs, OrSurface)
        rhs = maybe_parens(self.rhs, OrSurface)
        return f"{lhs} {rhs}"

    def children(self):
        return [self.lhs, self.rhs]

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.lhs, OrSurface)
        tokens += maybe_parens_tokens(self.rhs, OrSurface)
        return tokens

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [ConcatSurface(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [ConcatSurface(self.lhs, n) for n in nodes]
        else:
            return []

    def permutations(self):
        return get_all_trees(self)

    def over_approximation(self):
        lhs = self.lhs.over_approximation()
        if lhs is None:
            return None
        rhs = self.rhs.over_approximation()
        if rhs is None:
            return None
        return ConcatSurface(lhs, rhs)

    def under_approximation(self):
        lhs = self.lhs.under_approximation()
        if lhs is None:
            return None
        rhs = self.rhs.under_approximation()
        if rhs is None:
            return None
        return ConcatSurface(lhs, rhs)

    def unroll(self):
        return ConcatSurface(self.lhs.unroll(), self.rhs.unroll())

    def split(self):
        results = []
        for lhs in self.lhs.split():
            results.append(ConcatSurface(lhs, self.rhs))
        for rhs in self.rhs.split():
            results.append(ConcatSurface(self.lhs, rhs))
        return results


class OrSurface(Surface):
    _COGNITIVE_WEIGHT = CognitiveWeight.OR_SURFACE

    def __init__(self, lhs: Surface, rhs: Surface):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} | {self.rhs}"

    def children(self):
        return [self.lhs, self.rhs]

    def tokens(self):
        return [*self.lhs.tokens(), "|", *self.rhs.tokens()]

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [OrSurface(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [OrSurface(self.lhs, n) for n in nodes]
        else:
            return []

    def permutations(self):
        return get_all_trees(self)

    def over_approximation(self):
        lhs = self.lhs.over_approximation()
        rhs = self.rhs.over_approximation()
        if lhs is None:
            return rhs
        if rhs is None:
            return lhs
        return OrSurface(lhs, rhs)

    def under_approximation(self):
        lhs = self.lhs.under_approximation()
        rhs = self.rhs.under_approximation()
        if lhs is None:
            return rhs
        if rhs is None:
            return lhs
        return OrSurface(lhs, rhs)

    def unroll(self):
        return OrSurface(self.lhs.unroll(), self.rhs.unroll())

    def split(self):
        return self.lhs.split() + self.rhs.split()


class RepeatSurface(Surface):
    _COGNITIVE_WEIGHT = CognitiveWeight.REPEAT_SURFACE

    def __init__(self, surf: Surface, min: int, max: Optional[int]):
        self.surf = surf
        self.min = min
        self.max = max

    def __str__(self):
        surf = maybe_parens(self.surf, (ConcatSurface, OrSurface))
        quant = make_quantifier(self.min, self.max)
        return f"{surf}{quant}"

    def children(self):
        return [self.surf]

    def id_tuple(self):
        return super().id_tuple() + (self.min, self.max)

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.surf, (ConcatSurface, OrSurface))
        tokens += make_quantifier_tokens(self.min, self.max)
        return tokens

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        nodes = self.surf.expand_leftmost_hole(vocabularies, **kwargs)
        # avoid nesting repetitions
        nodes = [n for n in nodes if not isinstance(n, RepeatSurface)]
        return [RepeatSurface(n, self.min, self.max) for n in nodes]

    def permutations(self):
        return [RepeatSurface(p, self.min, self.max) for p in self.surf.permutations()]

    def over_approximation(self):
        if isinstance(self.surf, HoleSurface):
            return RepeatSurface(WildcardSurface(), self.min, self.max)
        surf = self.surf.over_approximation()
        if surf is None:
            return None
        return RepeatSurface(surf, self.min, self.max)

    def under_approximation(self):
        surf = self.surf.under_approximation()
        if surf is None:
            return None
        return RepeatSurface(surf, self.min, self.max)

    def unroll(self):
        if self.min <= 1 and self.max is None:
            return ConcatSurface(self.surf, ConcatSurface(self.surf, self))
        return self

    def num_quantifiers(self):
        return 1 + self.surf.num_quantifiers()


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

    def over_approximation(self):
        return RepeatTraversal(
            OrTraversal(IncomingWildcardTraversal(), OutgoingWildcardTraversal()),
            0,
            None,
        )

    def under_approximation(self):
        return None


class IncomingWildcardTraversal(Traversal):
    _COGNITIVE_WEIGHT = CognitiveWeight.INCOMING_WILDCARD_TRAVERSAL

    def __str__(self):
        return "<<"


class OutgoingWildcardTraversal(Traversal):
    _COGNITIVE_WEIGHT = CognitiveWeight.OUTGOING_WILDCARD_TRAVERSAL

    def __str__(self):
        return ">>"


class IncomingLabelTraversal(Traversal):
    _COGNITIVE_WEIGHT = CognitiveWeight.INCOMING_LABEL_TRAVERSAL

    def __init__(self, label: Matcher):
        self.label = label

    def __str__(self):
        return f"<{self.label}"

    def children(self):
        return [self.label]

    def tokens(self):
        return ["<"] + self.label.tokens()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.label.is_hole():
            return [
                IncomingLabelTraversal(ExactMatcher(v))
                for v in vocabularies.get(config.SYNTAX_FIELD, [])
            ]
        else:
            return []

    def over_approximation(self):
        label = self.label.over_approximation()
        if label is None:
            return None
        if isinstance(label, WildcardMatcher):
            return IncomingWildcardTraversal()
        return IncomingLabelTraversal(label)

    def under_approximation(self):
        label = self.label.under_approximation()
        if label is None:
            return None
        if isinstance(label, WildcardMatcher):
            return IncomingWildcardTraversal()
        return IncomingLabelTraversal(label)


class OutgoingLabelTraversal(Traversal):
    _COGNITIVE_WEIGHT = CognitiveWeight.OUTGOING_LABEL_TRAVERSAL

    def __init__(self, label: Matcher):
        self.label = label

    def __str__(self):
        return f">{self.label}"

    def children(self):
        return [self.label]

    def tokens(self):
        return [">"] + self.label.tokens()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.label.is_hole():
            return [
                OutgoingLabelTraversal(ExactMatcher(v))
                for v in vocabularies.get(config.SYNTAX_FIELD, [])
            ]
        else:
            return []

    def over_approximation(self):
        label = self.label.over_approximation()
        if label is None:
            return None
        if isinstance(label, WildcardMatcher):
            return OutgoingWildcardTraversal()
        return OutgoingLabelTraversal(label)

    def under_approximation(self):
        label = self.label.under_approximation()
        if label is None:
            return None
        if isinstance(label, WildcardMatcher):
            return OutgoingWildcardTraversal()
        return OutgoingLabelTraversal(label)


class ConcatTraversal(Traversal):
    _COGNITIVE_WEIGHT = CognitiveWeight.CONCAT_TRAVERSAL

    def __init__(self, lhs: Traversal, rhs: Traversal):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        lhs = maybe_parens(self.lhs, OrTraversal)
        rhs = maybe_parens(self.rhs, OrTraversal)
        return f"{lhs} {rhs}"

    def children(self):
        return [self.lhs, self.rhs]

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.lhs, OrTraversal)
        tokens += maybe_parens_tokens(self.rhs, OrTraversal)
        return tokens

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [ConcatTraversal(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [ConcatTraversal(self.lhs, n) for n in nodes]
        else:
            return []

    def permutations(self):
        return get_all_trees(self)

    def over_approximation(self):
        lhs = self.lhs.over_approximation()
        if lhs is None:
            return None
        rhs = self.rhs.over_approximation()
        if rhs is None:
            return None
        return ConcatTraversal(lhs, rhs)

    def under_approximation(self):
        lhs = self.lhs.under_approximation()
        if lhs is None:
            return None
        rhs = self.rhs.under_approximation()
        if rhs is None:
            return None
        return ConcatTraversal(lhs, rhs)

    def unroll(self):
        return ConcatTraversal(self.lhs.unroll(), self.rhs.unroll())

    def split(self):
        results = []
        for lhs in self.lhs.split():
            results.append(ConcatTraversal(lhs, self.rhs))
        for rhs in self.rhs.split():
            results.append(ConcatSurface(self.lhs, rhs))
        return results


class OrTraversal(Traversal):
    _COGNITIVE_WEIGHT = CognitiveWeight.OR_TRAVERSAL

    def __init__(self, lhs: Traversal, rhs: Traversal):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} | {self.rhs}"

    def children(self):
        return [self.lhs, self.rhs]

    def tokens(self):
        return self.lhs.tokens() + ["|"] + self.rhs.tokens()

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [OrTraversal(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies, **kwargs)
            return [OrTraversal(self.lhs, n) for n in nodes]
        else:
            return []

    def permutations(self):
        return get_all_trees(self)

    def over_approximation(self):
        lhs = self.lhs.over_approximation()
        rhs = self.rhs.over_approximation()
        if lhs is None:
            return rhs
        if rhs is None:
            return lhs
        return OrTraversal(lhs, rhs)

    def under_approximation(self):
        lhs = self.lhs.under_approximation()
        rhs = self.rhs.under_approximation()
        if lhs is None:
            return rhs
        if rhs is None:
            return lhs
        return OrTraversal(lhs, rhs)

    def unroll(self):
        return OrTraversal(self.lhs.unroll(), self.rhs.unroll())

    def split(self):
        return self.lhs.split() + self.rhs.split()


class RepeatTraversal(Traversal):
    _COGNITIVE_WEIGHT = CognitiveWeight.REPEAT_TRAVERSAL

    def __init__(self, traversal: Traversal, min: int, max: Optional[int]):
        self.traversal = traversal
        self.min = min
        self.max = max

    def __str__(self):
        traversal = maybe_parens(self.traversal, (ConcatTraversal, OrTraversal))
        quant = make_quantifier(self.min, self.max)
        return f"{traversal}{quant}"

    def children(self):
        return [self.traversal]

    def id_tuple(self):
        return super().id_tuple() + (self.min, self.max)

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.traversal, (ConcatTraversal, OrTraversal))
        tokens += make_quantifier_tokens(self.min, self.max)
        return tokens

    def expand_leftmost_hole(self, vocabularies, **kwargs):
        nodes = self.traversal.expand_leftmost_hole(vocabularies, **kwargs)
        nodes = [n for n in nodes if not isinstance(n, RepeatTraversal)]
        return [RepeatTraversal(n, self.min, self.max) for n in nodes]

    def permutations(self):
        return [
            RepeatTraversal(p, self.min, self.max)
            for p in self.traversal.permutations()
        ]

    def over_approximation(self):
        traversal = self.traversal.over_approximation()
        if traversal is None:
            return None
        return RepeatTraversal(traversal, self.min, self.max)

    def under_approximation(self):
        traversal = self.traversal.under_approximation()
        if traversal is None:
            return None
        return RepeatTraversal(traversal, self.min, self.max)

    def unroll(self):
        if self.min <= 1 and self.max is None:
            return ConcatTraversal(
                self.traversal, ConcatTraversal(self.traversal, self)
            )
        return self

    def num_quantifiers(self):
        return 1 + self.traversal.num_quantifiers()


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

    def over_approximation(self):
        raise NotImplementedError()

    def under_approximation(self):
        raise NotImplementedError()


class HybridQuery(Query):
    _COGNITIVE_WEIGHT = CognitiveWeight.HYBRID_QUERY

    def __init__(self, src: Surface, traversal: Traversal, dst: AstNode):
        self.src = src
        self.dst = dst
        self.traversal = traversal

    def __str__(self):
        src = maybe_parens(self.src, OrSurface)
        dst = maybe_parens(self.dst, OrSurface)
        traversal = maybe_parens(self.traversal, OrTraversal)
        return f"{src} {traversal} {dst}"

    def children(self):
        return [self.src, self.traversal, self.dst]

    def tokens(self):
        src = maybe_parens_tokens(self.src, OrSurface)
        dst = maybe_parens_tokens(self.dst, OrSurface)
        traversal = maybe_parens_tokens(self.traversal, OrTraversal)
        return src + traversal + dst

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

    def permutations(self):
        return [
            HybridQuery(src, traversal, dst)
            for src in self.src.permutations()
            for traversal in self.traversal.permutations()
            for dst in self.dst.permutations()
        ]

    def over_approximation(self):
        src = self.src.over_approximation()
        if src is None:
            return None
        traversal = self.traversal.over_approximation()
        if traversal is None:
            return None
        dst = self.dst.over_approximation()
        if dst is None:
            return None
        return HybridQuery(src, traversal, dst)

    def under_approximation(self):
        src = self.src.under_approximation()
        if src is None:
            return None
        traversal = self.traversal.under_approximation()
        if traversal is None:
            return None
        dst = self.dst.under_approximation()
        if dst is None:
            return None
        return HybridQuery(src, traversal, dst)

    def unroll(self):
        return HybridQuery(
            self.src.unroll(),
            self.traversal.unroll(),
            self.dst.unroll(),
        )

    def split(self):
        results = []
        for src in self.src.split():
            results.append(HybridQuery(src, self.traversal, self.dst))
        for traversal in self.traversal.split():
            results.append(HybridQuery(self.src, traversal, self.dst))
        for dst in self.dst.split():
            results.append(HybridQuery(self.src, self.traversal, dst))
        return results
