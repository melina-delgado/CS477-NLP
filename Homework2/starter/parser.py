import nltk
import itertools
from pandas import DataFrame

from collections import defaultdict
from nltk.ccg.chart import printCCGDerivation
from nltk.ccg.lexicon import Token

from utils import *

class CCGParser:

    """ Parse CCG according to the CKY algorithm. """

    DEFAULT_START = AtomicCategory("S")
    DEFAULT_RULE_SET = []

    def __init__(self, lexicon, rules=DEFAULT_RULE_SET):
        self.lexicon = lexicon
        self.rules = rules

    @staticmethod
    @rule(DEFAULT_RULE_SET, 2, "A")
    def application(cat1, cat2):
        """
        Implements the combination rule for function application.
        If cat1 and cat2 can be left or right-applied to each other (assuming cat1 is left of cat2),
        return the resulting category. Otherwise, return None.
        Hints:
            * isinstance(cat, CombinedCategory) tells you whether a category is combined or atomic.
            * For a combined category, cat.left and cat.right give the left and right subcategories.
            * For a combined category, cat.direction is either "\\" or "/".
            * You can use == to check whether categories are the same
        """
        # Cannot be equal
        if cat1 == cat2:
            return None

        # Cannot both be atomic 
        if isinstance(cat1, AtomicCategory) and isinstance(cat2, AtomicCategory):
            return None

        # Initialize
        cat1_comb = False
        cat2_comb = False

        if isinstance(cat1, CombinedCategory):
            cat1_left = cat1.left
            cat1_right = cat1.right
            cat1_dir = cat1.direction
            cat1_comb = True
        if isinstance(cat2, CombinedCategory):
            cat2_left = cat2.left
            cat2_right = cat2.right
            cat2_dir = cat2.direction
            cat2_comb = True

        # Cannot both be combined and opposing directions
        if cat1_comb and cat2_comb:
            if cat1_dir == "\\" and cat2_dir == "/":
                return None

        if cat1_comb:
            if cat1_dir == "/":
                if cat1_right == cat2:
                    return cat1_left

        if cat2_comb:
            if cat2_dir == "\\":
                if cat2_right == cat1:
                    return cat2_left

        return None
        raise NotImplementedError("application")

    @staticmethod
    @rule(DEFAULT_RULE_SET, 2, "C")
    def composition(cat1, cat2):
        """
        Implements the combination rule for function composition.
        If cat1 and cat2 can be left or right-composed, return the resulting category.
        Otherwise, return None.
        """

        if isinstance(cat1, AtomicCategory) or isinstance(cat2, AtomicCategory):
            return None
        
        # Right composition
        if cat1.right == cat2.left:
            if cat1.direction == "\\":
                return None

            cat3 = CombinedCategory(cat1.left, cat1.direction, cat2.right)

            return cat3

        # Left composition
        if cat1.left == cat2.right:
            if cat2.direction == "/":
                return None

            cat3 = CombinedCategory(cat2.left, cat2.direction, cat1.right)
            return cat3

        return None
        raise NotImplementedError("composition")

    @staticmethod
    @rule(DEFAULT_RULE_SET, 1, "T")
    def typeRaising(cat1, cat2):
        """
        Implements the combination rule for type raising.
        If cat2 satisfies the type-raising constraints, type-raise cat1 (and vice-versa).
        Return value when successful should be the tuple (cat, dir):
            * cat is the resulting category of the type-raising
            * dir is "/" or "\\" and represents the direction of the raising
            * If no type-raising is possible, return None
        Hint: use cat.innermostFunction() to implement the conditional checks described in the
            specification.
        """
        if isinstance(cat1, AtomicCategory) and isinstance(cat2, AtomicCategory):
            return None

        # Forward type-raising
        if isinstance(cat1, AtomicCategory):
            innermost = cat2.innermostFunction()
            if isinstance(innermost.right, CombinedCategory):
                return None
            if innermost.direction == "/":
                return None

            cat3 = CombinedCategory(innermost.left, "/", CombinedCategory(innermost.left, "\\", cat1))
            return (cat3, cat3.direction)

        # Backward type-raising
        if isinstance(cat2, AtomicCategory):
            innermost = cat1.innermostFunction()
            if isinstance(innermost.right, CombinedCategory):
                return None
            if innermost.direction == "\\":
                return None

            cat3 = CombinedCategory(innermost.left, "\\", CombinedCategory(innermost.left, "/", cat2))
            return (cat3, cat3.direction)

        return None
        raise NotImplementedError("typeRaising")

    class VocabException(Exception):
        pass

    def fillParseChart(self, tokens):
        """
        Builds and fills in a CKY parse chart for the sentence represented by tokens.
        The argument tokens is a list of words in the sentence.
        Each entry in the chart should be a list of Constituents:
            * Use AtomicConstituent(cat, word) to construct initialize Constituents of words.
            * Use CombinedConstituent(cat, leftPtr, rightPtr, rule) to construct Constituents
              produced by rules. leftPtr and rightPtr are the Constituent objects that combined to
              form the new Constituent, and rule should be the rule object itself.
        Should return (chart, parses), where parses is the final (top right) entry in the chart. 
        Each tuple in parses corresponds to a parse for the sentence.
        Hint: initialize the diagonal of the chart by looking up each token in the lexicon and then
            use self.rules to fill in the rest of the chart. Rules in self.rules are sorted by
            increasing arity (unary or binary), and you can use rule.arity to check the arity of a
            rule.
        """
        chart = defaultdict(list)
        
        for i, token in enumerate(tokens):
            for cat in self.lexicon.getCategories(token):
                chart[(i, i+1)].append(AtomicConstituent(cat, token))

        tok_len = len(tokens)
        for j in range(2, tok_len+1):
            for i in range(j-2, -1, -1):
                for k in range(i+1, j):
                    for con1 in chart[(i,k)]:
                        for con2 in chart[(k,j)]:
                            for rule in self.rules:
                                result = rule(con1.cat, con2.cat)
                                if result:
                                    if rule.arity == 1:
                                        cat3 = result[0]
                                        cat3_dir = result[1]
                                        if cat3_dir == "/":
                                            chart[(i,k)].append(CombinedConstituent(cat3,
                                                [con1], rule))
                                        else:
                                            chart[(k,j)].append(CombinedConstituent(cat3, [con2], rule))
                                    else:
                                        chart[(i,j)].append(CombinedConstituent(result, 
                                            [con1,con2], rule))
        return(chart, chart[(0, tok_len)])
        raise NotImplementedError("fillParseChart")

    @staticmethod
    def generateParseTree(cons, chart):
        """
        Helper function that returns an NLTK Tree object representing a parse.
        """
        token = Token(None, cons.cat, None)
        if isinstance(cons, AtomicConstituent):
            return nltk.tree.Tree(
                (token, u"Leaf"),
                [nltk.tree.Tree(token, [cons.word])]
            )
        else:
            if cons.rule == CCGParser.typeRaising:
                return nltk.tree.Tree(
                    (token, cons.rule.name),
                    [CCGParser.generateParseTree(cons.ptrs[0], chart)]
                )
            else:
                return nltk.tree.Tree(
                    (token, cons.rule.name),
                    [CCGParser.generateParseTree(cons.ptrs[0], chart),
                    CCGParser.generateParseTree(cons.ptrs[1], chart)]
                )

    def getParseTrees(self, tokens):
        """
        Reconstructs parse trees for the sentences by following backpointers.
        """
        chart, parses = self.fillParseChart(tokens)
        for cons in parses:
            yield CCGParser.generateParseTree(cons, chart)

    def accepts(self, tokens, sentCat=DEFAULT_START):
        """
        Return True iff the sentence represented by tokens is in this language (i.e. has at least
            one valid parse).
        """
        _, parses = self.fillParseChart(tokens)
        for cons in parses:
            if cons.cat == sentCat: return True
        return False
