.. _RelationBlockTutorial :

---------------
Relation Blocks
---------------

As stated in the :ref:`Movielens example <MovielensIndex>`, the complexity of
Bayesian FM is basically :math:`O(\mathrm{NNZ})`, that is, it grows
with the number of non-zero elements in the input sparse matrix.

This will in particular be troublesome when we include SVD++-like feature
into the feature matrix. In such a case, we include for each user the item ids
with which he or she had an interaction, and the complexity grows further by a factor of :math:`O(\mathrm{NNZ} / N_U)`.

However, we can get away with this catastrophic complexity if we notice the repeated pattern in the input matrix.
For the interested readers we refer `[Rendle, '13] <https://dl.acm.org/doi/abs/10.14778/2535573.2488340>`_ for the details
and `libFM Manual <http://www.libfm.org/libfm-1.40.manual.pdf>`_.

Below let us see how we can incorporate SVD++ - like feature efficiently using the relational data
again using Movielens 100K dataset.