Billion-scale similarity search with GPUs
Jeff Johnson
Matthijs Douze
Hervé Jégou
FacebookAlResearch
FacebookAlResearch
FacebookAlResearch
New York
Paris
Paris
2017
ABSTRACT
as the underlying processes either have high arithmetic com-
Similarity search finds application in specialized database
plexity and/or high data bandwidth demands [28], or cannot
beeffectivelypartitionedwithoutfailingduetocommuni-
systems handling complex data such as images or videos,
cation overhead or representation quality[38].Once pro-
whicharetypicallyrepresentedbyhigh-dimensionalfeatures
duced, their manipulation is itself arithmetically intensive.
and require specificindexing structures.This paper tackles
However, how to utilize GPU assets is not straightforward.
e
theproblem of better utilizing GPUsfor thistask.While
GPUs excel at data-parallel tasks, prior approaches are bot-
More generally, how to exploit new heterogeneous architec-
tures is a key subject for the database community [9].
tlenecked by algorithms that expose less parallelism, such as
8
k-min selection, or make poor use of the memory hierarchy.
In this context,searching by numerical similarity rather
than via structured relations is more suitable. This could be
We propose a design for k-selection that operates at up
to 55% of theoretical peak performance,enabling a nearest
to find the most similar content to a picture, or to find the
neighbor implementation that is 8.5xfaster than prior GPU
vectors thathave thehighest response toa linear classifier
on all vectors of a collection.
state of the art.We applyitindifferent similarity search
scenarios, by proposing optimized design for brute-force, ap-
One of the most expensive operations to be performed on
large collections is to compute a k-NNgraph.It is a directed
S
proximate and compressed-domain search based on product
graph where each vector of the database is a node and each
quantization. In all these setups, we outperform the state of
edge connects a node to its k nearest neighbors.This is
the artbylarge margins.Ourimplementation enablesthe
our fagship application. Note, state of the art methods like
construction of ahigh accuracyk-NNgraphon 95million
02.08734v
NN-Descent [15] have a large memory overhead on top of
images from the YFcc100M dataset in 35 minutes,and of
the datasetitself and cannotreadilyscaletothebillion-sized
a graph connecting 1 billion vectors in less than 12 hours
databases we consider.
on 4 Maxwell Titan X GPUs.We have open-sourced our
approach for the sake of comparison and reproducibility.
Such applicationsmustdeal withthe curseofdimension-
ality[46],renderingbothexhaustive search or exactindex-
ingfor non-exhaustive searchimpracticalonbillion-scale
1.INTRODUCTION
databases.Thisiswhythereisalargebodyofworkon
Images and videos constitute a new massive source of data
approximate search and/or graph construction. To handle
for indexing and search.Extensive metadata for this con-
huge datasets that donot fit inRAM,several approaches
tentis often not available.Search andinterpretation of this
employ an internal compressed representation of the vec-
and other human-generated content, like text, is difficult and
tors using an encoding.This is especially convenient for
important.A variety of machine learning and deep learn-
memory-limited devices like GPUs.Itturns out thataccept-
ing algorithms are being used to interpret and classify these
ing a minimal accuracy loss results in orders of magnitude
complex,real-world entities.Popular examples include the
of compression [21]. The most popular vector compression
text representation known as word2vec[32],representations
methods can be classified into either binary codes [18, 22],
of images by convolutional neural networks [39, 19], and im-
or quantization methods [25,37]. Both have the desirable
age descriptors forinstance search[20].Such representations
property that searching neighbors does not require recon-
or embeddings are usuallyreal-valued,high-dimensionalvec-
structing the vectors.
tors of 50 to 1000+ dimensions. Many of these vector repre-
Ourpaper focuses onmethodsbased onproductquanti-
sentations can only effectively be produced on GPU systems,
zation(PQ)codes,asthesewereshowntobemoreeffective
lhttps://github.com/facebookresearch/faiss
than binary codes [34].In addition,binary codes incur im-
portant overheads for non-exhaustive search methods [35].
Several improvements were proposed after the original prod-
uct quantization proposal known as IVFADC [25]; most are
difficult toimplement efficiently on GPU.For instance,the
inverted multi-index [4],useful for high-speed/low-quality
operating points, depends on a complicated“multi-sequence”
algorithm.The optimized product quantization or OPQ[17]
is alinear transformation ontheinputvectors thatimproves
the accuracy of the product quantization; it can be applied

as a pre-processing.The SIMD-optimized IVFADC imple-
Exact search.The exact solution computes thefullpair-
mentationfrom[2]operatesonlywithsub-optimalparame
ters (few coarse quantization centroids).Many other meth-
In practice, we use the decomposition
ods,like LOPQ and the Polysemous codes [27,16] are too
complextobe implementedefficiently on GPUs.
- yill2 = |;|l² + |lyll² - 2<c,yi).
(2)
There are many implementations of similarity search on
GPUs, but mostly with binary codes [36], small datasets [44],
The two first terms can be precomputed in one pass over
or exhaustive search [14, 40, 41]. To the best of our knowl-
the matrices X and Y whose rows are the [c§] and [yi]. The
edge, only the work by Wieschollek et al. [47] appears suit-
bottleneck is to evaluate (cj,yi>, equivalent to the matrix
able for billion-scale datasets with quantization codes. This
multiplication XY＇.The k-nearest neighbors for each of
is the prior state of the art on GPUs, which we compare
the nq queries are k-selected along each row of D.
against in Section 6.4.
Compressed-domainsearch.From now on,wefocus on
This paper makes the following contributions:
approximate nearest-neighbor search. We consider, in par-
ticular, the IVFADC' indexing structure [25]. The IVFADC
·a GPUk-selection algorithm,operating infastregister
index relies on two levels of quantization, and the database
memory and fexible enough tobe fusable with other
vectors are encoded. The database vector y is approximated
kernels, for which we provide a complexity analysis;
as:
·a near-optimal algorithmic layout for exact and ap-
y≈q(y)=q1(y)+q2(y-q1(y))
(3)
proximate k-nearest neighbor search on GPU;
where q1 : Rd → C1 C Rd and q2 : Rd → C2 C Rd are quan-
·a range of experiments that show that these improve-
tizers; i.e.,functions that output an element from a finite
set. Since the sets are finite, q(y) is encoded as the index of
q1(y) and that of q2(y -q1(y)). The first-level quantizer is a
mid- to large-scale nearest-neighbor search tasks,in
coarse quantizer and the second level fine quantizer encodes
single or multi-GPUconfigurations.
theresidualvector after thefirstlevel.
The paper is organized as follows. Section 2 introduces
The Asymmetric Distance Computation (ADC）search
thecontextand notation.Section3 reviews GPUarchi-
method returns an approximate result:
tecture and discusses problems appearing when using it for
LADc = k-argmin=o:ell - q(yi)Il2.
(4)
similarity search.Section 4 introduces one of our main con-
tributions,i.e.,our k-selection method for GPUs,while Sec-
For IVFADC the search is not exhaustive.Vectors for
tion5 provides details regarding the algorithm computation
which the distance is computed are pre-selected depending
layout. Finally, Section 6 provides extensive experiments for
on the first-level quantizer q1:
our approach, compares it to the state of the art, and shows
LIvF = T-argmincec, llc - cll2.
(5)
concreteuse cases for image collections.
The multi-probe parameter Tis the number of coarse-level
2.PROBLEM STATEMENT
centroids we consider.The quantizer operates a nearest-
neighbor search withexact distances,in the set of reproduc-
We are concernedwith similaritysearch invector collec-
tion values.Then,the IVFADC search computes
tions. Given the query vector c E Rd and the collection2
[yi]i=o:e (yi E Rd),we search:
LIVFADC =
k-argmin
α-q(yi)ll2.
(6)
L = k-argmin=o:ellc - yill2,
(1)
i=0:e s.t.q1(y)ELIVF
i.e.,we search the k nearest neighbors of c in terms of L2
Hence,IVFADCrelies on the same distance estimations as
distance. The L2 distance is used most often, as it is op-
the two-step quantization of ADC,but computes them only
timized by design when learning several embeddings (e.g.,
on a subset of vectors.
[20]), due to its attractive linear algebra properties.
The corresponding data structure,the inverted fle, groups
The lowest distances are collected by k-selection. For an
the vectors yi into |Ci| inverted lists Zi, .,ZicaI with homo-
array [ai]i=o:e, k-selection finds the k lowest valued elements
geneous q1(yi). Therefore, the most memory-intensive op-
[as]i=o:k, @s ≤ asi+1, along with the indices [si]i=o:k, 0 ≤
eration is computing LivFADc,and boils down to linearly
Si< l, of those elements from the input array. The a will be
scanning 7 inverted lists.
32-bit foating point values; the S are 32- or 64-bit integers.
The quantizers. The quantizers q1 and q2 have different
Other comparators are sometimes desired; e.g.，for cosine
properties.q1 needs to have a relatively low number of repro-
similarity we search for highest values.The order between
ductionvalues sothat thenumber of invertedlists does not
equivalent keys as: = as, is not specified.
explode. We typically use |Ci|  √e, trained via k-means.
Batching. Typically, searches are performed in batches
For q2, we can afford to spend more memory for a more ex-
of nq query vectors [αj]=o:nq (c, E Rd) in parallel, which
tensive representation. The ID of the vector (a 4- or 8-byte
allows for more fexibility when executing on multiple CPU
integer）is also stored in the inverted lists,so it makes no
threads or on GPU. Batching for k-selection entails selecting
sense to have shorter codes than that; i.e., log2 IC2| > 4 x 8.
nq × k elements and indices from nq separate arrays, where
Product quantizer. We use a product quantizer [25] for q2,
each array is of a potentiallydifferentlengthl≥k.
which provides a large number of reproductionvalues with-
2To avoid clutter in O-based indexing, we use the array no-
out increasing the processing_cost. It interprets the vector y
tation O :  to denote the range {O, ...,  - 1} inclusive.
as b sub-vectors y = [y°...yb-1], where b is an even divisor of
2

the dimension d.Each sub-vector is quantized with its own
opposed to 32, then only 1-2 other blocks can run concur-
quantizer, yielding the tuple (q(y°), .., q”-
qb-1(yb-1)). The
rentlyonthesameSM,resultinginlowoccupancy.Under
sub-quantizers typically have 256 reproduction values, to fit
high occupancy more blocks will be present across all SMs,
in one byte.The quantization value of the product quantizer
allowing more work to be in fight at once.
b-1
Memory types.Different blocks and kernels communicate
which from a storage point of view is just the concatena-
through global memory, typically 4-32 GB in size,with 5-
tion of thebytes produced by each sub-quantizer.Thus,the
10x higher bandwidth than CPU main memory.Shared
product quantizer generates b-byte codes with |C2|= 256b
reproduction values. The k-means dictionaries of the quan-
memory is analogous toCPU Ll cache in terms of speed.
tizers are small and quantization is computationally cheap.
GPU register file memory is the highest bandwidth memory.
Inordertomaintainthehighnumberofinstructionsinflight
on a GPU,a vast register file is also required:14 MB in the
3.GPU:OVERVIEWANDK-SELECTION
latest Pascal P10o,in contrast with a few tens of KB on
This section reviews salient details of Nvidia's general-
CPU.A ratio of 250:6.25:1 for register to shared to global
purpose GPU architecture and programming model [30]. We
memory aggregate cross-sectional bandwidth is typical on
thenfocus on one of theless GPU-compliantpartsinvolved
GPU, yielding 10-100s of TB/s for the register file [10].
in similarity search,namely the k-selection,and discuss the
3.2GPUregisterfile usage
literature and challenges.
3.1Architecture
Structured register data.Shared and register memory
usageinvolves efficiencytradeoffs;theylower occupancybut
GPU lanes and warps.The Nvidia GPU is a general-
can increase overall performance by retaining a larger work-
purpose computer that executes instruction streams using
ing set in a faster memory.Making heavy use of register-
a 32-wide vector of CUDAthreads(the warp);individual
resident data at the expense of occupancy or instead of
threads in the warp are referred to as lanes, with a lane
shared memory is often profitable[43].
ID from 0-31.Despite the “thread" terminology, the best
As the GPU register file is very large, storing structured
analogy tomodern vectorized multicore CPUs is thateach
data (not just temporary operands) is useful. A single lane
warp is a separate CPU hardware thread,as the warp shares
can use its (scalar) registers to solve a local task, but with
an instruction counter.Warp lanes taking different execu-
limited parallelism and storage. Instead, lanes in a GPU
tion paths results in warp divergence, reducing performance.
warp can instead exchange register data using the warp shuf-
Each lane has up to 255 32-bit registers in a shared register
fleinstruction,enabling warp-wide parallelism and storage.
file.The CPU analogy is that there are up to 255 vector
Lane-stride register array.A common pattern to achieve
registers of width 32,withwarplanes as SIMDvector lanes.
this is a lane-stride register array.That is,given elements
Collections of warps. A user-configurable collection of 1
[ai]i=o:e, each successive value is held in a register by neigh-
to32warpscomprisesablockoraco-operativethreadar-
boring lanes.The array is stored ine/32 registers per lane,
ray (CTA). Each block has a high speed shared memory, up
with & a multiple of 32.Lane j stores {aj,a32+j,.,ae-32+j},
while register r holds {a32r,a32r+1,.,a32r+31}.
to48KiBinsize.IndividualCUDAthreadshaveablock-
relative ID,called a thread id,which can be used to parti-
For manipulating the [ai], the register in which a is stored
tion and assign work. Each block is run on a single core of
(i.e.,[i/32]） and & must be known at assembly time, while
the GPU called a streaming multiprocessor (SM).Each SM
the lane (i.e.,i mod 32)can be runtime knowledge.A wide
has functional units,including ALUs,memory load/store
variety of access patterns(shift,any-to-any）are provided;
units,and various special instruction units. A GPU hides
we use the butterfy permutation[29]extensively.
execution latencies by having many operations in fight on
3.3 k-selection on CPU versus GPU
warps across all SMs. Each individual warp lane instruction
throughput is low and latency is high,but the aggregate
k-selection algorithms, often for arbitrarily large  and
arithmetic throughput of all SMs together is 5-10x higher
k，can be translatedto a GPU,including radic selection
than typical CPUs.
and bucket selection[1],probabilistic selection[33],quick-
select [14],and truncated sorts [40]. Their performance is
Grids and kernels. Blocks are organized in a grid of blocks
dominated by multiple passes over the input in global mem-
in a kernel.Each block is assigned a grid relative ID.The
ory. Sometimes for similarity search, the input distances are
kernel is the unit of work (instruction stream with argu-
computed on-the-fly or stored only in small blocks,not in
ments)scheduled by the host CPU for the GPU to execute.
their entirety. The full, explicit array might be too large to
After a block runs through to completion,new blocks can
fit into any memory, and its size could be unknown at the
be scheduled.Blocks from different kernels can run concur-
start of the processing, rendering algorithms that require
rently. Ordering between kernels is controllable via ordering
multiple passes impractical.They suffer from other issues
primitives such as streams and events.
as well.Quickselect requires partitioning on a storage of
size O(e),a data-dependent memory movement.This can
Resources and occupancy. The number of blocks execut-
resultin excessivememory transactions,orrequiringparallel
ingconcurrentlydependsuponsharedmemoryandregister
prefix sums to determine write offsets,with synchronization
resources used by each block. Per-CUDA thread register us-
overhead.Radix selection has no partitioning but multiple
age is determined at compilation time, while shared memory
passes are still required.
usage can be chosen at runtime. This usage affects occu-
pancy on the GPU. If a block demands all 48 KiB of shared
Heap parallelism.In similarity search applications, one
memoryfor itsprivate usage,or 128registers per thread as
is usually interested onlyin a small number of results,k<
3

1000 or so. In this regime, selection via max-heap is a typi-
Algorithm 1 Odd-size merging network
cal choice on the CPU,but heaps do not expose much data
function MERGE-ODD([Li=O:L,[Ri]i=0:R)
parallelism(due to serial tree update）and cannot saturate
parallel for i←O : min(lL,lR)do
SIMD execution units.The ad-heap[31]takes better advan-
> inverted 1st stage; inputs are already sorted
tage of parallelism available inheterogeneous systems,but
COMPARE-SWAP(LeL-i-1,Ri)
still attempts to partition serial and parallel work between
end for
appropriate execution units.Despite the serial nature of
parallel do
heapupdate,for small k the CPU can maintain allof its
>If lL = lr and a power-of-2,these are equivalent
state in the L1 cache with little effort, and L1 cache latency
MERGE-ODD-CONTINUE([Li}i=0:tL,left)
and bandwidth remains a limiting factor.Other similarity
MERGE-ODD-CONTINUE([Ri]i=O:R,right)
search components, like PQ code manipulation,tend tohave
enddo
greater impact on CPU performance [2].
end function
function MERGE-ODD-CONTINUE([c]=o:,p)
GPU heaps.Heaps can be similarly implemented on a
if l>1 then
GPU[7].However,a straightforward GPU heap implemen-
h←2[log2e]-1
largest power-of-2 <
tation suffers from high warp divergence and irregular, data-
parallel for i← 0 :l-h do
dependent memory movement, since the path taken for each
Implementedwith warp shufflebutterfly
inserted elementdependsupon other values in theheap.
COMPARE-SWAP(C,C+h)
GPU parallel priority queues[24]improve over the serial
end for
heap update by allowing multiple concurrent updates,but
parallel do
they require a potential number of small sorts for each insert
if p=left then
>left side recursion
and data-dependent memory movement. Moreover, it uses
MERGE-ODD-CONTINUE([ci]=0:-h,left)
multiple synchronization barriers through kernellaunches in
MERGE-ODD-CONTINUE([c=-h:,right)
different streams,plus the additional latency of successive
else
>right side recursion
kernellaunches and coordination with the CPUhost.
MERGE-ODD-CONTINUE([c]=0:h,left)
OthermorenovelGPUalgorithmsareavailablefor small
MERGE-ODD-CONTINUE([ci]i=h:,right)
k,namely the selection algorithm in the fgknn library [41].
end if
This is a complex algorithm that may suffer from too many
end do
synchronization points,greater kernel launch overhead,us-
end if
age of slower memories, excessive use of hierarchy, partition-
end function
ing and buffering. However, we take inspiration from this
particular algorithm through the use of parallel merges as
seen in their merge queuestructure.
Odd-size merging and sorting networks.If some input
4.FAST K-SELECTION ON THEGPU
data is already sorted,we can modify the network to avoid
merging steps. We may also not have a full power-of-2 set of
For anyCPUor GPU algorithm,either memory or arith-
data, in which case we can effciently shortcut to deal with
metic throughput should be the limiting factor as per the
the smaller size.
roofline performance model [48]. For input from global mem-
Algorithm 1 is an odd-sized merging network that merges
ory,k-selection cannot runfaster than the time required to
already sorted left and right arrays, each of arbitrary length.
scan theinputonce atpeakmemorybandwidth.We aim to
While the bitonic network merges bitonic sequences, we start
get as close to this limit as possible. Thus, we wish to per-
with monotonic sequences: sequences sorted monotonically.
form a single pass over the input data (from global memory
A bitonic merge is made monotonic by reversing the first
or produced on-the-fly, perhaps fused with a kernel that is
comparator stage.
generating the data).
The odd size algorithm is derived by considering arrays to
We want to keep intermediate state in the fastest memory:
be padded to the next highest power-of-2 size with dummy
the register file. The major disadvantage of register memory
is that the indexing into the register file must be known at
assembly time, which is a strong constraint on the algorithm.
13489
037
4.1In-register sorting
step 1
We use an in-register sorting primitive as a building block.
3430
987
Sorting networks are commonly used on SIMD architec-
step2
tures [13], as they exploit vector parallelism. They are eas-
789
ily implemented on the GPU, and we build sorting networks
with lane-stride register arrays.
step3
We use a variant of Batcher's bitonic sorting network [8],
1819
which is a set of parallel merges on an array of size 2k. Each
step 4
merge takes s arrays of length t (s and t a power of 2)to s/2
4
789
arrays of length 2t,using log2(t) parallel steps. A bitonic
sort applies this merge recursively: to sort an array of length
l, merge & arrays of length 1 to e/2 arrays of length 2, to e/4
Figure 1:Odd-size network merging arrays of sizes
arrays of length 4, successively to 1 sorted array of length ,
5and3.Bulletsindicate parallel compare/swap.
leading to (log2(e)² + log2(e)) parallel merge steps.
Dashed lines are elided elements or comparisons.
4

input
thread queue
warp queue
The elements (on the left of Figure 2) are processed in
ao
.····..·...
T.....T
laneo
groups of 32, the warp size. Lane j is responsible for pro-
insertion
.....
merging
M
lane 1
cessing {aj, a32+j,..); thus, if the elements come from global
memory, the reads are contiguous and coalesced into a min-
coalesced
network
imal number of memory transactions.
read
Data structures.Each lane j maintains a small queue
Wk-1
lane 31
of t elements in registers, called the thread queues [T?]i=o:t;
ordered from largest to smallest (Tj ≥ Ti+1). The choice of
t is made relative to k,see Section 4.3.The thread queue is
a frst-level filter for new values coming in. If a new a32i+j
Figure 2:Overview of WarpSelect.The input val-
ues stream in on the left, and the warp queue on the
is greater than the largest key currently in the queue, T?, it
is guaranteed that it won't be in the k smallest final results.
rightholdsthe outputresult.
The warp shares a lane-stride register array of k smallest
seen elements,[Wi]=o:k, called the warp queue. It is ordered
elements that are never swapped (the merge is monotonic)
from smallest to largest (W ≤ Wi+1); if the requested k is
and are already properly positioned; any comparisons with
not a multiple of 32, we round it up. This is a second level
dummy elements are elided.A left array is considered to
data structure thatwillbeused to maintain all of thek
be padded with dummy elements at the start; a right ar-
smallest warp-wide seen values. The thread and warp queues
ray has them at the end.A merge of two sorted arrays
are initialized to maximum sentinel values, e.g.,+oo.
of length lL and lr to a sorted array of &L + lR requires
Update. The three invariants maintained are:
[log2(max(lL,&r))]+1 parallel steps. Figure 1 shows Algo-
rithm 1's merging network for arrays of size 5 and 3, with 4
· all per-lane T? are not in the min-k
parallel steps.
The COMPARE-sWAP is implemented using warp shufles on
a lane-stride register array.Swaps with a stride a multiple
M
of 32occurdirectlywithin alane asthelaneholdsboth
elements locally. Swaps of stride ≤ 16 or a non-multiple of
·all aseen sofar in the min-k are contained in either
32 occur with warp shuffles. In practice, used array lengths
some lane's thread queue ([T]i=0:t,j=0:32), or in the
are multiples of 32 as they are held in lane-stride arrays.
warp queue.
Algorithm 2 Odd-size sorting network
Lane j receives a new a32i+j and attempts to insert it into
function SORT-ODD([c]i=0:e)
its thread queue. If a32i+j > T, then the new pair is by
if l>1 then
definition not in the k minimum, and can be rejected.
Otherwise,it is inserted into its proper sorted position
parallel do
SORT-ODD([α]i=0:[/2])
in the thread queue, thus ejecting the old T?. All lanes
SORT-ODD([ci]i=[e/2]:c)
complete doing this with their new received pair and their
thread queue, but it is now possible that the second invariant
end do
have been violated. Using the warp ballot instruction, we
MERGE-ODD([∞]=0:[e/2],[c]=[e/2]:e)
end if
determineifanylanehasviolatedthesecondinvariant.If
not,we are free to continue processing new elements.
end function
Restoring the invariants.If any lane has its invariant
Algorithm 2 extends the merge to a full sort.Assuming no
violated, then the warp uses ODD-MERGE to merge and sort
structure present in the input data, ([log2(e)]”+ [log2(e)])
the thread and warp queues together. The new warp queue
parallel steps are required for sorting data of length l.
4.2 WarpSelect
Algorithm 3 WARPSELECT pseudocode for lane j
Our k-selection implementation,WARPSELECT,maintains
function WARPSELECT(a)
state entirely in registers,requires only a single pass over
if a<Tthen
data and avoids cross-warp synchronization. It uses MERGE-
insert a into our [T]=0:t
ODD and sORT-ODD as primitives. Since the register file pro-
end if
vides much more storage than shared memory, it supports
if WARP-BALLOT(T < Wk-1) then
k ≤ 1024. Each warp is dedicated to k-selection to a single
 Reinterpret thread queues as lane-stride array
one of the n arrays [ai]. If n is large enough, a single warp
[α]=0:32t ← CAST([T/]=0:t,j=0:32)
per each [ai]will result in full GPU occupancy. Large & per
> concatenate and sort thread queues
warp is handled by recursive decomposition, if & is known in
SORT-ODD([Q]=0:32t)
advance.
MERGE-ODD([W]=0:k,[Qi=0:32t)
Overview. Our approach (Algorithm 3 and Figure 2) oper-
 Reinterpret lane-stride array as thread queues
[T]=0:t,j=0:32 ← CAST([Qi]=0:32t)
ates on values,with associated indices carried along(omit-
REVERSE-ARRAY([Ti]=0:t)
ted from the description for simplicity).It selects the k least
→Back in thread queue order,invariant restored
values that come from global memory, or from intermediate
end if
valueregisters iffusedinto anotherkernelprovidingtheval-
end function
ues. Let [ai]i=o:e be the sequence provided for selection.
5

will be the min-k elements across the merged, sorted queues,
own for exact nearest neighbor search in small datasets.It
and the new thread queues willbe the remainder,from min-
is also a component of many indexes in the literature.In
(k+1) to min-(k+32t+1). This restores the invariants and
our case, we use it for the IVFADC coarse quantizer q1.
we arefree to continueprocessing subsequentelements.
As stated in Section 2,the distance computation boils
Since the thread and warp queues are already sorted,we
down to a matrix multiplication.We use optimized GEMM
merge the sorted warp queue of length k with 32 sorted
routines in the cuBLAS library to calculate the -2(cj,yi)
arrays of length t. Supporting odd-sized merges is important
term forL2 distance,resulting in apartial distancematrix
because Batcher's formulation would require that 32t=k
D'. To complete the distance calculation, we use a fused
and is a power-of-2;thus if k = 1024,t must be 32.We
k-selection kernel that adds the |lyill² term to each entry of
found that the optimal t is way smaller (see below).
the distance matrix and immediately submits the value to
Using ODD-MERGE to merge the 32 already sorted thread
k-selection in registers. The Ilc,ll? term need not be taken
queues would require a struct-of-arrays toarray-of-structs
into account before k-selection.Kernel fusion thus allows
transposition inregisters across the warp, since thet succes-
for only 2 passes (GEMM write,k-select read) over D', com-
sive sorted values are held in different registers in the same
pared to other implementations that may require 3 or more.
lane rather than a lane-stride array. This is possible [12],
Row-wisek-selectionislikelynotfusablewithawell-tuned
but would use a comparable number of warp shufles, so we
GEMM kernel, or would result in lower overall efficiency.
just reinterpret the thread queue registers as an (unsorted)
As D' does not fit in GPU memory for realistic problem
lane-stride array and sort from scratch. Significant speedup
sizes, the problem is tiled over the batch of queries, with
is realizable by using ODD-MERGE for the merge of the ag-
tq ≤ nq queries being run in a single tile. Each of the 「nq/ta]
gregate sorted thread queues with the warp queue.
tiles are independent problems, but we run two in parallel
on different streams to better occupy the GPU, so the effec-
Handling theremainder.If there areremainder elements
tive memory requirementof Dis O(2ltq).The computation
because l is not a multiple of 32, those are inserted into the
can similarly be tiled over l. For very large input coming
thread queues forthelanes thathave them,after which we
from the CPU,we support buffering with pinned memory
proceed to the output stage.
to overlap CPU to GPU copy with GPU compute.
Output. A final sort and merge is made of the thread and
5.2IVFADC indexing
warp queues, after which the warp queue holds all min-k
values.
PQlookuptables.Atitscore,theIVFADCrequirescom-
4.3Complexity and parameter selection
puting the distance from a vector to a set of product quanti-
For each incoming group of 32 elements,WARPSELECT
zationreproductionvalues.By developingEquation(6)for
can perform 1, 2 or 3 constant-time operations, all happen-
a database vector y, we obtain:
ing in warp-wide parallel time:
llc - q(y)ll2 = I|x - q1(y) - q2(y - q1(y))I12.
(7)
1. read 32 elements, compare to all thread queue heads
T, cost C1, happens N1 times;
If we decompose the residual vectors left after q1 as:
2. if j ∈ {0,.., 31}, a32n+j < T, perform insertion sort
y-q1(y)= [yi..·y] and
(8)
on those specific thread queues, cost C2 = O(t),hap-
-q1(y）=[x1...x]
(9)
pens N2 times;
3. if 3j,T < Wk-1, sort and merge queues, cost C3 =
then the distance is rewritten as:
O(t log(32t)”+klog(max(k,32t))),happens N3 times.
Ilc - q(y)l2 = Ilx1 - q(y)12 + .. + Ilx - q(yb)2. (10)
Thus,the total cost is NiC1+ N2C2+ N3C3.N1=l/32
and on random data drawn independently, N2 = O(k log(e))
Each quantizer q',..,q has 256 reproduction values, so
when c and qi(y) are known all distances can be precom-
and N3 =O(klog(e)/t),see the Appendix for a full deriva-
puted and stored in tables Ti,...,T each of size 256 [25].
tion.Hence,the trade-off is to balance a cost in N2C2 and
Computing the sum (10） consists of b look-ups and addi-
oneinN3C3.Thepractical choicefor tgivenk and&was
madebyexperimenton avarietyofk-NNdata.Fork≤ 32.
tions.Comparing the cost to compute ndistances:
we use t=2,k≤ 128uses t=3,k≤ 256 uses t=4,and
·Explicit computation: n X d mutiply-adds;
k ≤ 1024 uses t = 8, all irrespective of l.
·With lookup tables: 256x d multiply-adds and n × b
5.COMPUTATIONLAYOUT
lookup-adds.
This section explains how IVFADC, one of the indexing
methods originally built upon product quantization [25],is
Thisis thekeytotheefficiency of theproduct quantizer.
implementedefficiently.Detailsondistancecomputations
In our GPU implementation,b is any multiple of 4 up to
andarticulationwithk-selection are thekeytounderstand-
64.The codes are stored as sequential groups of b bytes per
ing why this method can outperform more recent GPU-
vector within lists.
compliant approximate nearest neighbor strategies [47].
IVFADC lookup tables.When scanning over the ele-
5.1Exact search
ments of the inverted list LL (where by definition qi(y）is
Webrieflycomebacktothe exhaustive searchmethod,
constant),the look-up table method can be applied,as the
often referred to as exact brute-force.It is interesting on its
query r and qi(y) are known.
6

Moreover, the computation of the tables T1...T is fur-
for a single query,withk-selection fused with distance com-
ther optimized [5]. The expression of |α -q(y)]l2 in Equation
putation. This is possible as WARPSELECT does not fight for
(7) can be decomposed as:
the shared memory resource which is severely limited. This
reduces global memory write-back, since almost all interme-
la2(.)12 + 2<q(y),q2(..) + lxc - q1(y)12 2(x, q2(.)
diate results can be eliminated.However,unlike k-selection
term 1
term 2
term 3
overheadforexactcomputation,a significantportion of the
(11)
runtime is the gather from the T in shared memory and lin-
The objective is to minimize inner loop computations.
ear scanning of the Z from global memory;the write-back is
The computations we can do in advance and store in lookup
not a dominant contributor.Timing for the fused kernel is
tables are as follows:
improved by at most 15%, and for some problem sizes would
be subject to lower parallelism and worse performance with-
·Term 1 is independent of the query.It can be precom-
out subsequent decomposition.Therefore,and for reasons
puted from the quantizers,and stored in a table T of
of implementation simplicity, we do not use this layout.
size |Ci|× 256 × b;
·Term 2 is the distance to q1's reproduction value. It is
Algorithm4IVFPQbatchsearchroutine
thus a by-product of the first-level quantizer q1;
function IVFPQ-SEARCH([α1, ..., nq], Z1, ..,ZIc11)
for i←0:nq dobatch quantization of Section 5.1
·Term 3 can be computed independently of the inverted
LivF ←— T-argmincec, llα - cll2
list.Its computation costs d × 256 multiply-adds.
endfor
for i←0 : nq do
This decomposition is used to produce thelookup tables
L←
distance table
T1...T used during the scan of the inverted list.For a
Compute term 3 (see Section 5.2)
single query, computing the T x b tables from scratch costs
for L in LivF do
Tloops
T×d× 256multiply-adds,whilethisdecompositioncosts
Compute distance tables Ti, ..., Tb
256xd multiply-adds andxbx256 additions.OntheGPU,
the memory usage of T can be prohibitive, so we enable the
for j in L do
distance estimation,Equation (10)
decomposition onlywhenmemory is a not a concern.
d←l-q(y)l2
5.3GPUimplementation
Append (d,L,j) to L
end for
Algorithm 4 summarizes the process as one would im-
end for
plement it on a CPU. The inverted lists are stored as two
Ri←k-select smallest distances d from L
separate arrays, for PQ codes and associated IDs. IDs are
end for
resolvedonlyifk-selectiondeterminesk-nearestmember-
return R
ship. This lookup yields a few sparse memory reads in a
end function
large array, thus the IDs can optionally be stored on CPU
for tinyperformance cost.
List scanning.A kernel is responsible for scanning the T
5.4Multi-GPU parallelism
closest inverted lists for each query,and calculating the per-
Modern servers can support several GPUs. We employ
vector pair distances using thelookup tables T,.The T; are
this capability for both compute power and memory.
stored in shared memory:up to nqXTX max|Z|xb lookups
are required for a query set (trillions of accesses in practice),
Replication.If an index instance fits in the memory of a
and are random access.This limits b to at most 48(32-
single GPU,it can be replicated across R different GPUs.To
bit foating point）or 96(16-bit foating point）with current
query nq vectors, each replica handles a fraction nq/R of the
architectures.In case we donotuse the decomposition of
queries, joining the results back together on a single GPU
Equation (11),the T; are calculated by a separate kernel
or in CPU memory. Replication has near linear speedup,
before scanning.
except for a potential loss in efficiency for small nq·
Multi-pass kernels. Each nq X T pairs of query against
Sharding. If an index instance does not fit in the memory
inverted list can be processed independently.At one ex-
ofasingleGPU,anindexcanbeshardedacrossSdiffer-
treme, a block is dedicated to each of these, resulting in up
ent GPUs. For adding  vectors, each shard receives l/S of
to nq X T × maxi |Zi| partial results being written back to
the vectors, and for query, each shard handles the full query
global memory,which is then k-selected to nq × k final re-
set nq,joining the partial results (an additional round of k-
sults.Thisyieldshigh parallelism but can exceed available
selection is still required)on a single GPU or in CPU mem-
GPU global memory; as with exact search, we choose a tile
ory. For a given index size &, sharding will yield a speedup
size tq ≤ nq to reduce memory consumption,bounding its
(sharding has a query of nq against e/S versus replication
complexity by O(2tqT max; |Zil) with multi-streaming.
with a query of nq/R against e),but is usuallyless than
A single warp could be dedicated to k-selection of each
pure replication due to fixed overhead and cost of subse-
tq set of lists,which could result in low parallelism.We
quent k-selection.
introduce a two-pass k-selection, reducing tq × T × max; |Z|
Replication and sharding can be used together (S shards,
to tq × f × k partial results for some subdivision factor f.
each with R replicas for S x R GPUs in total). Sharding or
This is reduced again via k-selection to the final tqxk results.
replication are bothfairly trivial,and the same principle can
be used to distribute an index across multiple machines.
Fused kernel.As with exact search, we experimented with
a kernel that dedicates a single block to scanning all  lists
7

# centroids
method
# GPUs
256
4096
100
BIDMach [11]
1
320s
735s
Ours
1
140 s
316s
Ours
4
84s
100s
ms
10
untime
Table 1:MNIST8m k-means performance
truncatedbitonicsort
6.2 k-means clustering
fgknnselect
WarpSelect
The exact search method with k =1 can beused by a k-
memorybandwidthlimit
0.1
means clustering method in the assignment stage, to assign
1024
4096
16384
65536
nq training vectors to |Cilcentroids.Despite the fact that
arraylength
it does not use theIVFADC and k =1 selection is trivial (a
parallel reduction is used for the k = 1 case, not WARPSE-
Figure3:Runtimesfordifferentk-selectionmeth-
LECT),k-means is a good benchmark for the clustering used
ods,as a function of array length l.Simultaneous
to train the quantizer q1.
arrays processed are nq = 10000. k = 100 for full lines,
We apply the algorithm on MNIST8m images. The 8.1M
k = 1000 for dashed lines.
images are graylevel digits in 28x28 pixels, linearized to vec-
tors of 784-d.We compare this k-means implementation to
the GPU k-means of BIDMach [11],which was shown to be
6.EXPERIMENTS&APPLICATIONS
more efficient than several distributed k-means implemen-
This section compares our GPUk-selection and nearest-
tations that require dozens of machines?. Both algorithms
neighbor approach toexisting libraries.Unless stated other-
were run for 20 iterations. Table 1 shows that our imple-
wise,experiments are carried out on a 2x2.8GHz Intel Xeon
mentation is more than 2x faster,althoughboth are built
E5-2680v2with4MaxwellTitanXGPUsonCUDA8.0.
upon cuBLAS.Our implementation receives some benefit
from thek-selectionfusionintoL2distance computation.
6.1
k-selectionperformance
For multi-GPU execution via replicas, the speedup is close
We compare against two other GPU small k-selection im-
to linear for large enough problems (3.16x for 4 GPUs with
plementations:the row-based Merge Queue with Buffered
4096 centroids).Note that this benchmark is somewhat un-
Search and HierarchicalPartition extracted from the fgknn
realistic,as one would typically sub-sample the dataset ran-
library of Tang et al.[41]and Truncated Bitonic Sort(TBiS)
domlywhen sofew centroids arerequested.
from Sismanis et al.[40].Both were extracted from their re-
Large scale. We can also compare to [3], an approximate
spective exact searchlibraries.
CPU method that clusters 10° 128-d vectors to 85k cen-
We evaluatek-selection fork=100 and 1000 of eachrow
troids. Their clustering method runs in 46 minutes,but re-
from a row-major matrix nq × & of random 32-bit foating
quires 56minutes(atleast)of pre-processingto encode the
point values on a single Titan X. The batch size nq is fixed
vectors.Our method performs ecact k-means on 4 GPUs in
at 10000, and the array lengths  vary from 1000 to 128000.
52 minutes without any pre-processing.
Inputs and outputs to the problem remain resident in GPU
memory, with the output being of size nq × k,with corre-
6.3Exact nearest neighbor search
sponding indices. Thus, the input problem sizes range from
Weconsider a classicaldatasetusedtoevaluatenearest
40 MB(=1000)to 5.12GB((=128k).TBiS requireslarge
neighbor search: SiFT1M [25]. Its characteristic sizes are
auxiliary storage, and is limited to & ≤ 48000 in our tests.
( = 10°, d = 128, nq = 104. Computing the partial distance
Figure 3 shows our relative performance against TBiS and
matrix D' costs nq × & × d = 1.28 Tfop, which runs in less
fgknn.It alsoincludes the peak possible performance given
than one second on current GPUs.Figure 4 shows the cost
by the memory bandwidth limit of the Titan X. The rela-
of the distance computations against the cost of our tiling
tive performance of WARPSELECT over fgknn increases for
of the GEMM for the -2<cj,yi）term of Equation 2 and
larger k;even TBiS starts to outperform fgknn for larger 
the peak possiblek-selection performance on the distance
at k =1000.We look especially at thelargest &=128000.
matrix of size nq xl, which additionally accounts for reading
WARPSELECTis1.62xfasteratk=100,2.01×atk=1000.
the tiledresultmatrixD'atpeak memorybandwidth.
Performance against peak possible drops off for all imple-
In additionto our methodfrom Section 5,we include
mentations at larger k.WARPSELECT operates at 55% of
timesfromthe two GPUlibraries evaluatedfork-selection
peak at k = 100 but only 16% of peak at k = 1000.This
performance in Section 6.1. We make several observations:
is due to additional overhead assocated with bigger thread
queues and merge/sort networks for large k.
·for k-selection,the naive algorithm that sorts the full
result array for each query using thrust: :sort_by_key
Differencesfromfgknn.WARPSELECT isinfluencedby
is more than 10x slower than the comparison methods;
fgknn,but has several improvements:all state is maintained
in registers (no shared memory), no inter-warp synchroniza-
·L2 distance and k-selection cost is dominant for all but
tion or buffering is used, no “hierarchical partition",the k-
our method,whichhas 85%of thepeak possible
selection canbe fused into other kernels, and it uses odd-size
performance,assuming GEMM usage and our tiling
networks for efficient merging and sorting.
3BIDMach numbers from https://github.com/BIDData/
BIDMach/wiki/Benchmarks#KMeans
8

120
-2xySGEMM (as tiled)
4 TitanX:m=64,S=1,R=4
3.5
peakpossiblek-select
100
4 TitanX:m=32,S=1,R=4
ourmethod
3
4 TitanX:m=16,S=1,R=4
truncatedbitonicsort
fgknn
80
2.5
runtime
2
60
40
20
0.5
YFCC100M
0
4
16
64
256
1024
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
k
10-intersection at 10
24
Figure 4:Exact search k-NN time for the SIFT1M
4 Titan X:m=40,S=4, R=1
(hours)
dataset with varying k on 1 Titan X GPU.
20
4 TitanX:m=20,S=2,R=2
8M40:m=40,S=4,R=2
8M40:m=20,S=2,R=4
aw pnq yden NN-Yy
16
of the partial distance matrix D' on top of GEMM is
close to optimal.The cuBLASGEMM itselfhaslow
12
efficiency for small reduction sizes (d = 128);
8
·Our fused L2/k-selectionkernel is important.Our
same exact algorithm without fusion (requiring an ad-
3
中
DEEP1B
ditional pass through D') is at least 25% slower.
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Efficientk-selection is even more important in situations
10-intersection at10
where approximate methods are used to compute distances,
because the relative cost of k-selection with respect to dis-
Figure 5:Speed/accuracy trade-off of brute-force
tance computationincreases.
10-NN graph construction for theYFCC1ooM and
6.4Billion-scale approximate search
DEEP1B datasets.
There are few studies on GPU-based approximate nearest-
neighbor search on large datasets (l >> 10°). We report a
different,it shows that making searches on GPUs is a game-
few comparison points here on index search, using standard
changer in terms of speed achievable on a single machine.
datasets and evaluation protocol in thisfield.
6.5The k-NN graph
SIFTiM.For the sake of completeness, we first compare
our GPU search speed on SiFT1M with the implementation
An example usage of our similarity search method is to
of Wieschollek et al. [47]. They obtain a nearest neighbor re-
construct a k-nearest neighbor graph of a dataset via brute
call at1(fraction of queries where the true nearestneighbor
force (all vectors queried against the entire index).
is in the top 1 result）of R@1=0.51,and R@100=0.86 in
Experimentalsetup.Weevaluatethetrade-offbetween
0.02 ms per query on a Titan X. For the same time budget,
speed,precision and memory on two datasets:95 million
our implementation obtains R@1=0.80 and R@100=0.95.
images from the YFcc100M dataset [42] and DEEP1B. For
SIFTiB. We compare again with Wieschollek et al., on the
YFCc100M, we compute CNN descriptors as the one-before-
SIFT1B dataset [26] of 1 billion SIFT image features at nq =
last layer of a ResNet [23], reduced to d=128 with PCA.
104. We compare the search performance in terms of same
The evaluation measures the trade-off between:
memory usage for similar accuracy (more accurate methods
·Speed:How much time it takes to build the IVFADC
may involve greater search time or memory usage). On a
index from scratch and construct the whole k-NN graph
single GPU,with m = 8 bytes per vector, R@10=0.376 in
(k = 10) by searching nearest neighbors for all vectors
17.7 μs per query vector, versus their reported R@10 = 0.35
in the dataset.Thus,this is an end-to-end test that
in 150 μs per query vector. Thus, our implementation is
includes indexing as well as search time;
more accurate at a speed8.5xfaster.
●Quality: We sample 10,000 images for which we com-
DEEP1B. We also experimented on the DEEP1B dataset [6]
pute the exact nearest neighbors. Our accuracy mea-
of &=1 billion CNN representations for images at nq = 104.
sure is the fraction of 10 found nearest neighbors that
The paper that introduces the dataset reports CPU results
are within the ground-truth 10 nearest neighbors.
(1 thread): R@1=0.45 in 20 ms search time per vector.We
use a PQ encoding of m = 20, with d = 80 via OPQ [17],
For YFCc100M, we use a coarse quantizer (216 centroids),
and |Ci| = 218, which uses a comparable dataset storage as
and consider m =16,32 and 64 byte PQ encodings for each
the original paper (20 GB). This requires multiple GPUs as
vector. For DEEP1B, we pre-process the vectors to d = 120
it is too large for a single GPU's global memory, so we con-
via OPQ, use |Cil = 218 and consider m = 20, 40. For a
sider 4 GPUs with S = 2,R = 2. We obtain a R@1=0.4517
given encoding,we vary  from 1 to 256,to obtain trade-
in 0.0133msper vector.While thehardware platforms are
offs between efficiency and quality, as seen in Figure 5.
9

Figure 6:Path in the k-NN graph of 95 million images from YFCC1ooM. The first and the last image are
given;thealgorithmcomputesthesmoothestpathbetweenthem.
Discussion.ForYFCc100MweusedS=1,R=4.An
7.CONCLUSION
accuracy of more than 0.8 is obtained in 35 minutes.For
The arithmetic throughput and memory bandwidth of
DEEP1B,a lower-quality graph can be built in 6 hours,
GPUs are well into the teraflops and hundreds of gigabytes
with higher quality in about half a day.We also experi-
per second. However, implementing algorithms that ap-
mented with more GPUs by doubling the replica set, us-
proach these performance levels is complex and counter-
ing 8 Maxwell M40s (the M40 is roughly equivalent in per-
intuitive. In this paper, we presented the algorithmic struc-
formance to the Titan X).Performance is improved sub-
ture of similarity search methods that achieves near-optimal
performance on GPUs.
For comparison,thelargest k-NNgraph construction we
This work enables applications that needed complex ap-
areawareofusedadatasetcomprising 36.5million384-
proximate algorithms before. For example, the approaches
d vectors,which took a cluster of 128 CPU servers 108.7
presented here make it possible to do exact k-means cluster-
hours of compute [45], using NN-Descent [15]. Note that
ing or to compute thek-NNgraphwith simplebrute-force
NN-Descent could also build or refine the k-NNgraph for
approaches in less time than a CPU (or a cluster of them)
the datasets we consider,but it has a large memory over-
would take to do this approximately.
head over the graph storage, which is already 80 GB for
GPU hardware is now very common on scientific work-
DEEP1B. Moreover it requires random access across all vec-
stations, due to their popularity for machine learning algo-
tors (384GB for DEEP1B)
rithms.Webelieve that our workfurther demonstratestheir
The largest GPU k-NN graph construction we found is a
interest for database applications. Along with this work, we
brute-force construction using exact search with GEMM, of
are publishing a carefully engineered implementation of this
a dataset of 20 million 15,000-d vectors,which took a cluster
paper's algorithms, so that these GPUs can now also be used
of 32 Tesla C2050 GPUs 10 days [14].Assuming computa-
for efficient similarity search.
tion scales with GEMM cost for the distance matrix,this
approach for DEEP1B would take an impractical 200 days
of computation time on their cluster.
8.REFERENCES
[1]T.Alabi, J.D. Blanchard,B.Gordon, and R.Steinbach.
6.6Using the k-NN graph
Fast k-selection algorithms for graphics processing units.
ACMJournalofErperimentalAlgorithmics,
When a k-NN graph has been constructed for an image
17:4.2:4.14.2:4.29, October 2012.
dataset,we can find paths in the graph between any two
[2]F.André,A.-M.Kermarrec,and N.L.Scouarnec. Cache
images, provided there is a single connected component (this
locality is not enough: High-performance nearest neighbor
is the case).For example,we can search the shortest path
search with product quantization fast scan. In Proc.
between two images of fowers, by propagating neighbors
International Conference on Very Large DataBases,pages
from a starting image to a destination image. Denoting by
288-299,2015.
S and D the source and destination images, and dij the
[3]Y.Avrithis, Y. Kalantidis,E.Anagnostopoulos, and I. Z.
Emiris. Web-scale image clustering revisited. In Proc.
distance between nodes, we search the path P = {p1,., Pn}
International Conference on Computer Vision,pages
with p1 = S and pn = D such that
1502-1510,2015.
[4]A. Babenko and V. Lempitsky. The inverted multi-index.
min max dpipi+1,
(12)
In Proc.IEEE Conference on ComputerVisionand
Pi=1..n
Pattern Recognition, pages 3069-3076,June 2012.
[5]A.Babenko and V.Lempitsky.Improving bilayer product
i.e.,we want to favor smooth transitions.An example re-
quantization for billion-scale approximate nearest neighbors
sult is shown in Figure 6 from YFcc100M4.It was ob-
in high dimensions.arXiv preprint arXiu:1404.1831,2014.
tained after 20 seconds of propagation in a k-NN graph with
[6]A.Babenko and V. Lempitsky.Efficient indexing of
billion-scale datasets of deep descriptors.In Proc.IEEE
k =15 neighbors.Since there are many flower images in the
Conference onComputerVision andPatternRecognition,
dataset,the transitions are smooth.
pages 2055-2063, June 2016.
[7]R.Barrientos,J.Gomez, C. Tenllado,M.Prieto,and
M. Marin. knn query processing in metric spaces using
4The mapping from vectors to images is not available for
GPUs.InInternationalEuropeanConferenceonParallel
DEEP1B
andDistributedComputing,volume 6852of LectureNotes
10

in Computer Science,pages 380-392,Bordeaux,France,
and Signal Processing, pages 861-864,May 2011.
September 2011.Springer.
[27]Y.Kalantidis and Y.Avrithis.Locally optimized product
[8]K.E.Batcher.Sorting networks and their applications. In
quantizationfor approximate nearest neighbor search.In
Proc.Spring Joint Computer Conference,AFIPS'68
Proc.IEEEConferenceonComputerVisionandPattern
(Spring),pages 307-314,New York,NY,USA,1968.ACM.
Recognition,pages 2329-2336,June 2014.
[9] P. Boncz,W. Lehner, and T. Neumann. Special issue:
[28] A. Krizhevsky, 1. Sutskever, and G. E. Hinton. Imagenet
Modernhardware.The VLDB Journal, 25(5):623-624,
classificationwithdeepconvolutionalneuralnetworks.In
2016.
AdvancesinNeuralInformation ProcessingSystems,pages
[10]J. Canny, D.L.W.Hall, and D.Klein.A multi-teraflop
10971105, 2012.
constituency parser using GPUs.InProc.Empirical
[29]F.T.Leighton.IntroductiontoParallelAlgorithmsand
Methods on Natural Language Processing,pages 1898-1907.
Architectures: Array,Trees,Hypercubes.Morgan
ACL,2013.
Kaufmann Publishers Inc.,San Francisco, CA,USA,1992.
[11]J.Canny and H. Zhao.Bidmach:Large-scale learning with
[30] E. Lindholm, J. Nickolls, S. Oberman, and J. Montrym.
zero memory allocation.InBigLearn workshop,NIPS,
NVIDIA Tesla:a unified graphics and computing
2013.
architecture.IEEE Micro,28(2):39-55,March 2008.
[12]B.Catanzaro,A.Keller,and M.Garland.A decomposition
[31]W.LiuandB.Vinter.Ad-heap:Anefficientheapdata
for in-place matrix transposition.In Proc.ACM
structure for asymmetric multicore processors.In Proc.of
Symposium onPrinciples andPracticeofParallel
Workshop on General Purpose Processing Using GPUs,
Programming,PPoPP"14,pages 193-206,2014.
pages 54:54-54:63.ACM, 2014.
[13] J. Chhugani, A. D. Nguyen, V. W. Lee, W. Macy,
[32]T.Mikolov, I. Sutskever,K.Chen,G.S.Corrado, and
M. Hagog, Y.-K. Chen, A. Baransi, S. Kumar, and
J. Dean.Distributed representations of words and phrases
P. Dubey. Efficient implementation of sorting on multi-core
andtheir compositionality.In Advances inNeural
simd cpu architecture.Proc.VLDB Endow.,
Information Processing Systems,pages 3111-3119, 2013.
1(2):1313-1324,August 2008.
[33]L. Monroe,J. Wendelberger, and S. Michalak. Randomized
[14]A.Dashti.Efficient computation of k-nearest neighbor
selection on the GPU.InProc.ACMSymposium onHigh
graphs for large high-dimensional data sets on gpu clusters.
Performance Graphics,pages 89-98,2011.
Master's thesis,University of Wisconsin Milwaukee,August
[34]M. Norouzi and D. Fleet. Cartesian k-means. In Proc.
2013.
IEEEConferenceonComputerVisionandPattern
[15] W. Dong, M. Charikar, and K. Li. Efficient k-nearest
Recognition, pages 3017-3024, June 2013.
neighbor graph construction for generic similaritymeasures.
[35]M. Norouzi, A.Punjani, and D.J. Fleet. Fast search in
InWWW:Proceeding of theInternational Conference on
Hamming space with multi-index hashing. In Proc. IEEE
World Wide Web, pages 577-586, March 2011.
ConferenceonComputerVisionandPatternRecognition,
[16] M. Douze,H. Jégou, and F.Perronnin. Polysemous codes.
pages 3108-3115, 2012.
In Proc.European Conference on Computer Vision,pages
[36]J.Pan and D. Manocha.Fast GPU-based locality sensitive
785-801.Springer,October 2016.
hashing for k-nearest neighbor computation.In Proc.ACM
[17]T.Ge,K.He,Q.Ke,and J. Sun. Optimized product
InternationalConferenceonAdvancesinGeographic
quantization.IEEE Trans.PAMI, 36(4):744-755,2014.
Information Systems,pages 211-220,2011.
[18]Y.Gong and S.Lazebnik.Iterative quantization: A
[37]L.Paulevé, H.Jégou, and L.Amsaleg.Locality sensitive
procrustean approach to learning binary codes. In Proc.
hashing: a comparison of hash function types and querying
IEEEConferenceonComputerVisionandPattern
mechanisms.Pattern recognition letters,31(11):1348-1358,
Recognition,pages 817-824,June 2011.
August 2010.
[19]Y.Gong,L.Wang, R.Guo, and S.Lazebnik.Multi-scale
[38]O.Shamir.Fundamental limits of online and distributed
orderlesspoolingof deep convolutional activationfeatures.
algorithmsfor statisticallearning andestimation.In
In Proc.European Conference on Computer Vision,pages
AdvancesinNeuralInformation Processing Systems,pages
392-407, 2014.
163-171, 2014.
[20]A.Gordo,J.Almazan,J.Revaud, and D.Larlus.Deep
[39]A.Sharif Razavian,H.Azizpour,J.Sullivan,and
image retrieval:Learning global representations for image
S. Carlsson. CNN features off-the-shelf: an astounding
search.In Proc.European Conference on Computer Vision,
baseline for recognition.In CVPR workshops, pages
pages 241-257, 2016.
512-519,2014.
[21]S.Han, H. Mao,and W. J. Dally. Deep compression:
[40] N. Sismanis, N. Pitsianis, and X. Sun. Parallel search of
Compressing deep neural networks with pruning, trained
k-nearest neighbors with synchronous operations. In IEEE
quantization and huffman coding. arXiv preprint
High Performance Ertreme Computing Conference,pages
arXiv:1510.00149,2015.
1-6, 2012.
[22]K.He,F.Wen,and J.Sun.K-means hashing:An
[41] X. Tang, Z. Huang, D. M. Eyers, S.Mills, and M. Guo.
affinity-preservingquantizationmethodforlearningbinary
Efficient selection algorithm for fast k-nn search on GPUs.
compact codes.In Proc.IEEE Conference on Computer
InIEEEInternationalParallel&DistributedProcessing
Visionand Pattern Recognition,pages 2938-2945,June
Symposium,pages 397-406,2015.
2013.
[42]B.Thomee,D.A.Shamma,G.Friedland,B.Elizalde,
[23]K.He,X. Zhang,S.Ren, and J. Sun. Deep residual
K.Ni,D.Poland,D.Borth,and L.-J.Li.YFCC100M:The
learning for image recognition. In Proc. IEEE Conference
new data in multimediaresearch.Communications of the
on ComputerVision and PatternRecognition,pages
ACM,59(2):64-73,January2016.
770-778,June 2016.
[43]V.Volkov and J.W.Demmel.Benchmarking GPUs to tune
[24]X.He,D.Agarwal, and S.K.Prasad. Design and
dense linear algebra.In Proc.ACM/IEEE Conference on
implementation of a parallel priority queue on many-core
Supercomputing, pages 31:1-31:11, 2008.
architectures.IEEEInternationalConference onHigh
[44]A.Wakatani andA.Murakami. GPGPU implementation of
Performance Computing, pages 1-10, 2012.
nearest neighbor search with product quantization.In
[25]H.Jégou,M.Douze,and C.Schmid.Product quantization
IEEEInternationalSymposiumonParallelandDistributed
for nearest neighbor search.IEEETrans.PAMI,
Processing with Applications,pages 248-253,2014.
33(1):117-128, January 2011.
[45]T.Warashina,K.Aoyama, H. Sawada,and T.Hattori.
[26] H. Jégou, R. Tavenard, M. Douze,and L.Amsaleg.
Efficient k-nearest neighbor graph construction using
Searching in onebillionvectors:re-rankwithsource
mapreduce for large-scale data sets. IEICE Transactions,
coding.In International Conference on Acoustics,Speech,
11

97-D(12):3142-3154, 2014.
The last case is the probability of: there is a & - 1 se-
[46]R. Weber, H.-J. Schek, and S.Blott.A quantitative
quence with m - 1 successive min-k elements preceding us,
analysis andperformance studyfor similarity-search
and the current element is in the successive min-k,or the
methods in high-dimensional spaces. In Proc. International
current element is not in the successive min-k,m ones are
Conference on Very Large DataBases,pages 194-205,1998.
before us.We can then develop a recurrence relationship for
[47]P. Wieschollek, O. Wang, A. Sorkine-Hornung, and
π(l,k,t,1).Note that
H.P. A.Lensch.Efficient large-scale approximate nearest
neighbor search on the GPU.In Proc. IEEE Conference on
Computer Vision and Pattern Recognition,pages
min((bt+max(0,t-1)),)
2027-2035,June 2016.
8(e,b,k,t) :=
M
r(l,m,k)
(17)
[48]S.Williams,A.Waterman,and D. Patterson.Roofline: An
insightful visual performance model for multicore
m=bt
architectures.Communications of the ACM, 52(4):65-76,
for b where O≤ bt ≤ is the fraction of all sequences of
April 2009.
length C that will force b sorts of data by winning the thread
Appendix: Complexity analysis of WARPSELECT
queue ballot, as there have to be bt to (bt + max(0,t - 1))
elements in the successive min-k for these sorts to happen (as
We derive the average number of times updates are triggered
the min-k elements will overflow the threadqueues).There
in WARPSELECT,for use in Section 4.3.
are at most [e/t」 won ballots that can occur,as it takes t
Let the input to k-selection be a sequence {a1,a2, .., ae}
separate sequential current min-k seen elements to win the
(1-based indexing),a randomly chosen permutation of a set
ballot. π(l,k,t,1) is thus the expectation of this over all
of distinct elements.Elements are read sequentially in C
possible b:
groups of size w (the warp; in our case, w = 32); assume 
is a multiple of w,so c = l/w.Recall that t is the thread
[e/t]
queue length. We call elements prior to or at position n
π(e,k,t,1) =b· o(e,b,k,t).
(18)
in the min-k seen so far the successive min-k (at n).The
b=1
likelihood that an is in the successive min-k at n is:
This can be computed by dynamic programming. Analyti-
ifn≤k
cally, note that for t = 1, k = 1, π(e,1,1,1) is the harmonic
α(n,k) :
(13)
k/n
ifn>k
(the Euler-Mascheroni constant ) as &→ ∞o.
as each an,n >k has a k/n chance as all permutations are
equally likely, and all elements in the first k qualify.
or O(klog(e)),as the first k elements are in the successive
Counting the insertion sorts. In a given lane, an inser-
min-k,and the expectation for the rest is
tion sort is triggered if theincomingvalueis in thesuccessive
values, where co is the previous won warp ballot. The prob-
D,k≤ D≤of successive min-k determinations Dmade
ability of this happening is:
for each possible {a1, .., ae}. The number of won ballots for
k+t
α(wco +(c-co),k+t)≈
for c > k.
each case is by definition[D/t],as the thread queue must
(14)
wc
fill up t times. Thus, π(e, k,t, 1) = O(k log(e)/t).
The approximation considers that the thread queue has seen
Multiplelanes.The w>1 case is complicated by the
all the wc values, not just those assigned to its lane. The
fact that there are joint probabilities to consider (if more
probability of any lane triggering an insertion sort is then:
than one of the w workers triggers a sort for a given group,
k+t
only one sort takes place).However,the likelihood can be
(15)
bounded. Let π'(e,k,t, w) be the expected won ballots as-
C
suming no mutual interference between the w workers for
Here the approximation is a first-order Taylor expansion.
winning ballots (i.e.,we win b ballots if there are b ≤ w
Summing up the probabilities over c gives an expected num-
workers that independently win a ballot at a single step),
ber of insertions of N2 ≈ (k + t) log(c) = O(k log(e/w)).
but withthe shared min-k set after each sort from thejoint
Counting full sorts.We seek N3 = π(l,k,t,w),the ex-
sequence.Assume that k≥w.Then:
pected number of full sorts required for WARPSELECT.
Single lane. For now, we assume w = 1, so c = l. Let
[e/w]—[k/w]
T'(l,k, 1,w) ≤w
k
(l, m,k) be the probability that in an sequence {a1,...,ae},
M
(+[m/y])m
exactly m of the elements as encountered by a sequential
i=1
scanner (w = 1) are in the successive min-k. Given m, there
≤wπ([e/w],k,1,1) =O(wklog(e/w))
(m) places where these successive min-k elements can
are
(19)
occur.It is given by a recurrence relation:
where the likelihood of the w workers seeing a successive
min-k element has an upper bound of that of the first worker
l=0 and m = 0
at each step.As before,the number of won ballots is scaled
0
l=0and m>0
by t, so π'(e,k,t,w)= O(wk log(e/w)/t).Mutual interfer-
r(l,m,k) :=
ence can only reduce the number of ballots, so we obtain the
0
e>0andm=0
same upper bound for π(e, k,t, w).
(r(e - 1,m - 1,k)· α(e,k)+
((e-1,m,k)·(1 -α(l,k)))otherwise.
(16)
12