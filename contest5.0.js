/**
 * @param {number} n
 * @return {number}
 */
var minOperations = function (n) {
    return Math.floor(n * n / 4);
};

/**
 * @param {string} s
 * @return {boolean}
 */
var halvesAreAlike = function (s) {
    const isVowel = (char) => /[aeiuo]/i.test(char);
    const countVowels = (str) => str.split('').reduce((total, item) => {
        if (isVowel(item)) {
            total++
        }

        return total;
    }, 0);
    const first = s.slice(0, s.length / 2);
    const second = s.slice(s.length / 2);

    return countVowels(first) === countVowels(second)
};

/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @param {TreeNode} p
 * @return {TreeNode}
 */
var inorderSuccessor = function (root, p) {
    const stack = [];
    let curr = root, isPassed = false;
    while (curr || stack.length) {
        while (curr) {
            stack.push(curr);
            curr = curr.left;
        }
        curr = stack.pop();
        if (isPassed) {
            return curr;
        }
        if (curr === p) {
            isPassed = true;
        }
        curr = curr.right;
    }
    return null;
};

/**
 * @param {string} digits
 * @return {string[]}
 */
const letterCombinations = function (digits) {
    if (!digits.length) return [];

    let obj = {
        2: ['a', 'b', 'c'],
        3: ['d', 'e', 'f'],
        4: ['g', 'h', 'i'],
        5: ['j', 'k', 'l'],
        6: ['m', 'n', 'o'],
        7: ['p', 'q', 'r', 's'],
        8: ['t', 'u', 'v'],
        9: ['w', 'x', 'y', 'z'],
    };

    let result = [];
    const helperRecursion = str => {
        if (str.length == 0) {
            return;
        } else {
            let tempArr = [];
            if (result.length === 0) {
                tempArr = obj[str[0]];
            } else {
                const alphasArray = obj[str[0]];
                const resultLength = result.length;
                for (let i = 0; i < alphasArray.length; i++) {
                    for (let j = 0; j < resultLength; j++) {
                        tempArr.push(result[j] + alphasArray[i]);
                    }
                }
            }
            result = tempArr;
            helperRecursion(str.substring(1));
        }
    }
    helperRecursion(digits);
    return result;
};

/**
 * @param {number[][]} matrix
 * @return {number}
 */
var longestIncreasingPath = function (M) {
    let ylen = M.length, xlen = M[0].length, ans = 0,
        memo = Array.from({length: ylen}, el => new Uint16Array(xlen))
    const dfs = (y, x) => {
        if (memo[y][x]) return memo[y][x]
        let val = M[y][x]
        memo[y][x] = 1 + Math.max(
            y < ylen - 1 && M[y + 1][x] < val ? dfs(y + 1, x) : 0,
            y > 0 && M[y - 1][x] < val ? dfs(y - 1, x) : 0,
            x < xlen - 1 && M[y][x + 1] < val ? dfs(y, x + 1) : 0,
            x > 0 && M[y][x - 1] < val ? dfs(y, x - 1) : 0)
        return memo[y][x]
    }
    for (let i = 0; i < ylen; i++)
        for (let j = 0; j < xlen; j++)
            ans = Math.max(ans, dfs(i, j))
    return ans
};

/**
 * @param {number} n
 * @param {number} k
 * @return {number[]}
 */
var constructArray = function (n, k) {
    let ans = [], low = 1, high = k + 1

    for (let i = 1; i <= n; i++) {
        if (i <= k + 1) {
            ans[i - 1] = (i - 1) % 2 === 0 ? low++ : high--
        } else {
            ans[i - 1] = i
        }
    }

    return ans
}

/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * function NestedInteger() {
 *
 *     Return true if this NestedInteger holds a single integer, rather than a nested list.
 *     @return {boolean}
 *     this.isInteger = function() {
 *         ...
 *     };
 *
 *     Return the single integer that this NestedInteger holds, if it holds a single integer
 *     Return null if this NestedInteger holds a nested list
 *     @return {integer}
 *     this.getInteger = function() {
 *         ...
 *     };
 *
 *     Return the nested list that this NestedInteger holds, if it holds a nested list
 *     Return null if this NestedInteger holds a single integer
 *     @return {NestedInteger[]}
 *     this.getList = function() {
 *         ...
 *     };
 * };
 */
/**
 * @constructor
 * @param {NestedInteger[]} nestedList
 */
// idea: retrieve of all data into array then pop one by one
function NestedIterator(nestedList) {
    let d = [];
    for (const e of nestedList) {
        if (e.isInteger()) {
            d.push(e.getInteger());
        } else {
            dfs(e);
        }
    }
    return {
        hasNext,
        next
    }

    function dfs(input) {
        if (input.isInteger()) return d.push(input.getInteger());
        let list = input.getList();
        for (const e of list) {
            dfs(e);
        }
    }

    function hasNext() {
        return d.length > 0;
    }

    function next() {
        return d.shift();
    }
};

/**
 * Your NestedIterator will be called like this:
 * var i = new NestedIterator(nestedList), a = [];
 * while (i.hasNext()) a.push(i.next());
 */

/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} head
 * @param {number} x
 * @return {ListNode}
 */
var partition = function (head, x) {
    let lessDummy = new ListNode(-1);
    let greaterDummy = new ListNode(-1);
    let current = head;
    let less = lessDummy;
    let greater = greaterDummy;
    while (current) {
        if (current.val < x) {
            less.next = current;
            less = current;
        } else if (current.val >= x) {
            greater.next = current;
            greater = current;
        }
        current = current.next;
    }
    greater.next = null;
    less.next = greaterDummy.next;
    return lessDummy.next;
};

/**
 * @param {number[]} data
 * @return {number}
 */
var minSwaps = function (data) {
    const ones = data.filter(d => d).length;
    let k = 0;
    for (let i = 0; i < ones; i += 1) k += data[i];
    let ans = ones - k;
    for (let i = ones; i < data.length; i += 1) {
        k += data[i] - data[i - ones];
        ans = Math.min(ans, ones - k);
    }
    return ans;
};

/**
 * @param {string} s
 * @param {number} k
 * @return {string}
 */
var removeDuplicates = function (s, k) {
    let rdStk = [[s[0], 1]]; //s string is at least size 1 so init our stack
    for (let i = 1; i < s.length; i++) {
        if (rdStk.length && s[i] === rdStk[rdStk.length - 1][0]) { //is current letter same as top of stack?
            if (++rdStk[rdStk.length - 1][1] === k) rdStk.pop();
        } else {
            rdStk.push([s[i], 1]);
        }
    }
    return rdStk.reduce((str, elm) => str + elm[0].repeat(elm[1]), '');
};

/**
 * @param {number[][]} matrix
 * @param {number} target
 * @return {number}
 */
var numSubmatrixSumTarget = function (matrix, target) {
    let row = matrix.length, col = matrix[0].length;
    for (let i = 1; i < row; i++) {
        for (let j = 0; j < col; j++) {
            matrix[i][j] += matrix[i - 1][j];
        }
    }
    let ans = 0;
    for (let i = 0; i < row; i++) {
        for (let j = i; j < row; j++) {
            let map = new Map(), sum = 0;
            map.set(0, 1);
            for (let k = 0; k < col; k++) {
                sum += matrix[j][k] - (i != 0 ? matrix[i - 1][k] : 0);
                ans += (map.get(sum - target) || 0);
                map.set(sum, (map.get(sum) || 0) + 1);
            }
        }
    }
    return ans;
};

/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} head
 * @param {number} n
 * @return {ListNode}
 */
var removeNthFromEnd = function (head, n) {
    let temp = new ListNode(0);
    temp.next = head;
    let slow = temp,
        fast = temp;

    while (fast.next !== null) {
        fast = fast.next
        if (n-- <= 0) {
            slow = slow.next;
        }
    }
    slow.next = slow.next.next;
    return temp.next;
};

/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
var combinationSum4 = function (nums, target, counter = 0, map = {}) {
    if (counter === nums.length) {
        return 0;
    }

    if (map[target] >= 0)
        return map[target];

    if (target === 0) {
        return 1;
    } else if (target < 0) {
        return 0;
    }

    let output = 0;
    let tCounter = 0;
    while (tCounter < nums.length) {
        output += combinationSum4(nums, target - nums[tCounter], tCounter, map);
        tCounter++;
    }
    map[target] = output;
    return output;
};

/**
 * // Definition for a Node.
 * function Node(val, children) {
 *    this.val = val;
 *    this.children = children;
 * };
 */

/**
 * @param {Node} root
 * @return {number[]}
 */
var preorder = function (root) {
    if (!root) return []
    var stack = [root];
    var result = [];
    while (stack.length) {
        var node = stack.pop();
        result.push(node.val);
        while (node.children.length) {
            stack.push(node.children.pop());
        }
    }
    return result;
};

/**
 * @param {number[][]} triangle
 * @return {number}
 */
var minimumTotal = function (triangle) {
    for (let i = triangle.length - 2; ~i; i--) {
        for (let j = 0; j < triangle.length; j++) {
            triangle[i][j] += Math.min(triangle[i + 1][j], triangle[i + 1][j + 1]);
        }
    }
    return triangle[0][0];
};

/**
 * @param {number[]} arr
 * @return {number}
 */
var missingNumber = function (arr) {
    let left = 0;
    let right = arr.length - 1;
    let mid;
    let diff = (arr[right] - arr[left]) / arr.length;
    while (left < right) {
        mid = Math.floor((right + left) / 2);
        if (arr[mid] == (arr[0] + mid * diff)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return arr[0] + diff * left;
};

/**
 * @param {string} s
 * @return {number}
 */
var countBinarySubstrings = function (s) {
    let prevRunLength = 0;
    let currRunLength = 1;
    let res = 0;

    for (let i = 1; i < s.length; i += 1) {
        if (s[i - 1] === s[i]) {
            currRunLength += 1;
        } else {
            prevRunLength = currRunLength;
            currRunLength = 1;
        }
        // must be a substring if length of prev 0s or 1s >= curr 0s or 1s
        if (prevRunLength >= currRunLength) {
            res += 1;
        }
    }

    return res;
};

/**
 * @param {number} n
 * @param {number[][]} connections
 * @return {number[][]}
 */
var criticalConnections = function (n, connections) {
    let graph = Array.from(Array(n), () => []); //Graph that will store edges of each node
    let results = [];
    let levels = []; //Holds the min level of a node reachable in a cycle
    for (let pair of connections) { //Convert connection tuples to a graph with edges
        graph[pair[0]].push(pair[1]);
        graph[pair[1]].push(pair[0]);
    }
    let dfs = (parent, curr, level) => {
        levels[curr] = level + 1;
        let currLevel = levels[curr];
        for (let next of graph[curr]) {
            if (next === parent) continue;
            if (!levels[next]) dfs(curr, next, level + 1); //Acts as a "visited" array
            levels[curr] = Math.min(levels[curr], levels[next]); //Update current node to the min value reachable
            if (currLevel < levels[next]) { //Not able to reach a node with lesser cycle value a.k.a no cycle
                results.push([curr, next]); //Critical connection
            }
        }
    };
    dfs(-1, 0, 0);
    return results;
};

/**
 * @param {number[][]} matrix
 * @return {void} Do not return anything, modify matrix in-place instead.
 */
var rotate = function (matrix) {
    // Firstly Transpose The Matrix
    for (let i = 0; i < matrix.length; i++) {
        for (let j = i; j < matrix.length; j++) {
            var Temp = matrix[j][i]
            matrix[j][i] = matrix[i][j]
            matrix[i][j] = Temp
        }
    }
    // Secondly Make Reflected Image Of Matrix
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix.length / 2; j++) {
            var Temp = matrix[i][j]
            matrix[i][j] = matrix[i][matrix.length - j - 1]
            matrix[i][matrix.length - j - 1] = Temp
        }
    }
};

/**
 * @param {number[]} heights
 * @param {number} bricks
 * @param {number} ladders
 * @return {number}
 */
var furthestBuilding = function (heights, bricks, ladders) {
    if (ladders >= heights.length - 1)
        return heights.length - 1;

    const dp = new Array(heights.length).fill(-Infinity);
    dp[0] = bricks;

    let res = 0, diff;
    for (let i = 0; i <= ladders; i++) {
        let pre = dp[0];

        for (let j = 1; j < heights.length; j++) {
            if (heights[j] <= heights[j - 1]) {
                [pre, dp[j]] = [dp[j], dp[j - 1]];
            } else {
                diff = heights[j] - heights[j - 1];
                let temp = dp[j];

                if (i === 0)
                    dp[j] = dp[j - 1] - diff;
                else
                    dp[j] = Math.max(dp[j - 1] - diff, pre);

                if (dp[j] < 0)
                    break;

                pre = temp;

            }
            res = Math.max(j, res);
        }

        // Corner case: if we have enough remaining ladders to achieve the end of buildings, return the last index
        if (res + (ladders - i) >= heights.length - 1)
            return heights.length - 1;
    }


    return res;
};

/**
 * @param {number[][]} obstacleGrid
 * @return {number}
 */
var uniquePathsWithObstacles = function (obstacleGrid) {
    const max_row = obstacleGrid.length;
    const max_col = obstacleGrid[0].length;
    const memo = new Array(max_row).fill().map(() => new Array(max_col));

    const backtrack = (row, col) => {
        // Base Case: When row/col is out of bounds or when an obstacle is met
        if (row < 0 || col < 0 || row >= max_row || col >= max_col || obstacleGrid[row][col] === 1) {
            return 0;
        }

        // If we have a record saved in memo, return that to prevent duplicate recursive calls
        if (memo[row][col] != undefined) return memo[row][col];

        // If we reached the goal
        if (row === max_row - 1 && col === max_col - 1) return 1;

        // Memoize the sum of moving down or right 1 space
        memo[row][col] = backtrack(row + 1, col) + backtrack(row, col + 1);

        return memo[row][col];
    }

    return backtrack(0, 0);
};

/**
 * @param {number[][]} slots1
 * @param {number[][]} slots2
 * @param {number} duration
 * @return {number[]}
 */
var minAvailableDuration = function (slots1, slots2, duration) {

    slots1.sort((a, b) => a[0] - b[0])
    slots2.sort((a, b) => a[0] - b[0])

    let s1 = 0, s2 = 0;

    while (s1 < slots1.length && s2 < slots2.length) {
        const [b1, e1] = slots1[s1]
        const [b2, e2] = slots2[s2]

        const start = Math.max(b1, b2)
        const end = Math.min(e1, e2);

        if (end - start >= duration) return [start, start + duration]

        if (e1 < e2) s1++;
        else s2++;
    }
    return []
};

/**
 * @param {number} x
 * @param {number} y
 * @param {number} bound
 * @return {number[]}
 */
var powerfulIntegers = function (x, y, bound) {
    const result = new Set();
    const xPowMax = (x > 1) ? Math.round(Math.log(bound) / Math.log(x)) : 1;
    const yPowMax = (y > 1) ? Math.round(Math.log(bound) / Math.log(y)) : 1;

    for (let i = 0; i <= xPowMax; i++) {
        for (let j = 0; j <= yPowMax; j++) {
            const candidate = Math.pow(x, i) + Math.pow(y, j);
            if (candidate <= bound) {
                result.add(candidate)
            }
        }
    }

    return [...result];
};

/**
 * @param {string[]} words
 */
var WordFilter = function (words) {
    this.dictionary = new Map();

    words.forEach((word, indexWord) => {
        let prefixString = '';
        for (let prefix = 0; prefix < word.length; prefix++) {
            prefixString += word[prefix];

            for (let suffix = 0; suffix < word.length; suffix++) {
                this.dictionary.set(`${prefixString}-${word.substr(suffix, word.length)}`, indexWord);
            }
        }
    });


};

/**
 * @param {string} prefix
 * @param {string} suffix
 * @return {number}
 */
WordFilter.prototype.f = function (prefix, suffix) {
    const key = `${prefix}-${suffix}`;

    return this.dictionary.has(key) ? this.dictionary.get(key) : -1;
};

/**
 * @param {number[][]} courses
 * @return {number}
 */
var scheduleCourse = function (courses) {
    if (!courses.length) return 0
    // think it backwards, course with the last end date shall be taken last
    // so we sort with bigger d first
    courses.sort(([t1, d1], [t2, d2]) => d2 - d1)
    // keep note of date and course count, index=count,value = empty before day
    let arr = new Array(courses.length + 1).fill(-1)
    arr[0] = Number.MAX_SAFE_INTEGER, maxCount = 0
    // loop course
    for (let [t, d] of courses) {
        // we update from [maxCount+1] first based on maxCount, then smaller counts
        for (let i = maxCount + 1; i > 0; i--) {
            // if smaller count has enough left days for this course, then arrange it
            arr[i] = Math.max(arr[i], Math.min(arr[i - 1], d) - t)
            // if we can arrange this course, then we got a new max count
            if (arr[i] > -1 && i > maxCount) maxCount = i
        }
    }
    return maxCount
};
/**
 * @param {number[]} nums
 * @return {number[]}
 */
var runningSum = function (nums) {
    return nums.reduce((total, item, index) => {
        const prev = total[index - 1];

        if (prev !== undefined) {
            total.push(item + prev);
        } else {
            total.push(item);
        }

        return total;
    }, []);
};

/**
 * @param {number[]} nums
 * @return {boolean}
 */
var checkPossibility = function (nums) {
    let count = 0;
    for (let i = 1; i < nums.length; i++) {
        if (nums[i] < nums[i - 1]) {
            if (nums[i] < nums[i - 2]) {
                nums[i] = nums[i - 1];
            }
            count++;
        }
        if (count > 1) {
            return false;
        }
    }
    return true;
};

/**
 * @param {number[]} nums
 * @return {number}
 */
var jump = function (nums) {
    let arr = Array.from({length: nums.length}).fill(0)
    let goal = nums.length - 1

    for (let i = 0; i < nums.length; i++) {

        let maxIndex = Math.min(goal, i + nums[i])
        for (let j = maxIndex; j > i; j--) {
            if (!arr[j]) {
                arr[j] = arr[i] + 1
            } else {
                break
            }
        }

        if (arr[goal]) break
    }

    return arr.pop()
};

/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {ListNode} head
 * @return {TreeNode}
 */
var sortedListToBST = function (head) {

    if (head === null) {
        return null;
    }

    const list = [];
    while (head !== null) {
        list.push(head.val);
        head = head.next;
    }

    return findHead(0, list.length - 1);

    function findHead(start, end) {

        if (start > end) {
            return null;
        }
        const middle = Math.ceil((start + end) / 2);

        const left = findHead(start, middle - 1);
        const right = findHead(middle + 1, end);

        return new TreeNode(list[middle], left, right);
    }
};

/**
 * @param {string} word1
 * @param {string} word2
 * @return {number}
 */
var minDistance = function (word1, word2) {
    const n1 = word1.length, n2 = word2.length;
    let f = new Array(n1 + 1).fill(0).map(x => Array(n2 + 1).fill(5000))

    for (let i = 0; i < n1 + 1; i++) {
        f[i][0] = i;
    }
    for (let i = 0; i < n2 + 1; i++) {
        f[0][i] = i
    }
    // console.log(f)

    for (let i = 1; i < n1 + 1; i++) {
        for (let j = 1; j < n2 + 1; j++) {
            f[i][j] = Math.min(f[i - 1][j], f[i][j - 1]) + 1
            if (word1[i - 1] === word2[j - 1]) {
                f[i][j] = Math.min(f[i][j], f[i - 1][j - 1])
            }
        }
    }

    // console.log(f)
    return f[n1][n2]
};

/**
 * @param {string} left
 * @param {string} right
 * @return {number}
 */
var superpalindromesInRange = function (left, right) {
    let ans = 9 >= left && 9 <= right ? 1 : 0

    const isPal = str => {
        for (let i = 0, j = str.length - 1; i < j; i++, j--)
            if (str.charAt(i) !== str.charAt(j)) return false
        return true
    }

    for (let i = 1; i < 19684; i++) {
        let num = i.toString(3)
        if (isPal(num)) {
            let square = BigInt(num) * BigInt(num)
            if (square > right) return ans
            if (square >= left && isPal(square.toString())) ans++
        }
    }
    return ans
};

/**
 * @param {number[]} target
 * @return {boolean}
 */
var isPossible = function (T) {
    if (T.length === 1 && T[0] !== 1) return false
    let sum = T.reduce((a, b) => a + b)
    T.sort((a, b) => b - a)

    while (sum !== T.length) {
        let m = T[0] - (sum - T[0]) * (Math.trunc(T[0] / (sum - T[0]) - 1) || 1);
        [sum, T[0]] = [sum - T[0] + m, m]
        if (T[0] < 1) return false
        for (let i = 0; T[i] < T[i + 1]; i++) [T[i], T[i + 1]] = [T[i + 1], T[i]]
    }

    return true
};

/**
 * @param {number} n
 * @return {number}
 */
var countPrimes = function (n) {
    if (n < 2) return 0;

    const sieve = new Array(n + 1).fill(true);

    for (let i = 2; i <= Math.sqrt(n); i++) {
        if (sieve[i]) {
            for (let j = i * i; j <= n; j += i) {
                sieve[j] = false;
            }
        }
    }

    let count = 0;

    for (let i = 2; i < n; i++) {
        if (sieve[i] == true) count++;
    }

    return count;
};

/**
 * @param {number[]} cardPoints
 * @param {number} k
 * @return {number}
 */
var maxScore = function (cardPoints, k) {
    let windowSize = cardPoints.length - k, arrSum = 0, minWindowSum = Number.MAX_SAFE_INTEGER, windowSum = 0;
    for (let i = 0; i < cardPoints.length; i++) {
        arrSum += cardPoints[i];
        if (i <= windowSize - 1) {//If the window size is less then or equal to the expected size, then just keep adding the new element to the window
            windowSum += cardPoints[i];
        } else {//If window size is full then add the latest element and remove the oldest element from the window
            windowSum += cardPoints[i];
            windowSum -= cardPoints[i - windowSize];
        }
        if (i >= windowSize - 1) {//If the window size is equal to the expected size then check if this window is having the minimun sum
            minWindowSum = Math.min(minWindowSum, windowSum);
        }
    }
    return arrSum - minWindowSum;
};

/**
 * @param {number[][]} matrix
 */
var NumMatrix = function (matrix) {
    let n = matrix.length
    if (n == 0) return
    let m = matrix[0].length

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
            let sum = 0
            if (i > 0) sum += matrix[i - 1][j]
            if (j > 0) sum += matrix[i][j - 1]
            if (i > 0 && j > 0) sum -= matrix[i - 1][j - 1]
            matrix[i][j] += sum
        }
    }
    this.matrix = matrix
};

NumMatrix.prototype.sumRegion = function (row1, col1, row2, col2) {
    let matrix = this.matrix
    let res = matrix[row2][col2]

    if (row1 > 0) res -= matrix[row1 - 1][col2]
    if (col1 > 0) res -= matrix[row2][col1 - 1]
    if (row1 > 0 && col1 > 0) res += matrix[row1 - 1][col1 - 1]

    return res
};

/**
 * @param {string} s
 * @return {string[]}
 */
const ambiguousCoordinates = function (s) {

    const ans = [];

    // outer for loop to create all combinations of numbers on left and right side
    for (let i = 2; i < s.length - 1; i++) {
        let lft = s.slice(1, i);
        let rght = s.slice(i, s.length - 1);

        //  add combinations with no decimals
        const ls = [lft];
        const rs = [rght];

        // inner for loop to find all combinations of left side with a decimal
        for (let i = 1; i < lft.length; i++) {
            temp1 = lft.slice(0, i);
            temp2 = lft.slice(i);
            ls.push(`${temp1}.${temp2}`);
        }

        // inner for loop to find all combinations of right side with a decimal
        for (let i = 1; i < rght.length; i++) {
            temp1 = rght.slice(0, i);
            temp2 = rght.slice(i, rght.length);
            rs.push(`${temp1}.${temp2}`);
        }

        // push in all valid answers that pass the regex
        for (let l of ls) {
            if (!/^([1-9]\d*|0\.\d*[1-9]|[1-9]\d*\.\d*[1-9]|0)$/i.test(l)) continue;
            for (let r of rs) {
                if (!/^([1-9]\d*|0\.\d*[1-9]|[1-9]\d*\.\d*[1-9]|0)$/i.test(r)) continue;
                ans.push(`(${l}, ${r})`);
            }
        }
    }
    return ans;
};

/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {void} Do not return anything, modify root in-place instead.
 */
var flatten = function (root) {
    function binaryPreOrderTraversal(root) {
        // pre-order
        let values = [root.val];

        if (root.left) {
            values = values.concat(binaryPreOrderTraversal(root.left));
        }
        if (root.right) {
            values = values.concat(binaryPreOrderTraversal(root.right));
        }

        return values;
    }

    if (!root) {
        return [];
    }


    const data = binaryPreOrderTraversal(root).slice(1);
    let prev = root;
    for (let elem of data) {
        let node = new TreeNode(elem);
        prev.left = null;
        prev.right = node;
        prev = node;
    }
};

const stat1 = [{
    startDate: '9:00',
    endDate: '10:20',
    present: true
}, {
    startDate: '10:20',
    endDate: '10:30',
    present: false
}, {
    startDate: '10:30',
    endDate: '11:45',
    present: true
}, {
    startDate: '11:45',
    endDate: '13:15',
    present: false
}, {
    startDate: '13:15',
    endDate: '14:00',
    present: true
}, {
    startDate: '14:00',
    endDate: '15:00',
    present: false
}, {
    startDate: '15:00',
    endDate: '18:00',
    present: true
}];

const stat2 = [{
    startDate: '9:55',
    endDate: '10:15',
    present: true
}, {
    startDate: '10:15',
    endDate: '14:00',
    present: false
}, {
    startDate: '14:00',
    endDate: '15:20',
    present: true
}, {
    startDate: '15:20',
    endDate: '17:00',
    present: false
}, {
    startDate: '17:00',
    endDate: '18:20',
    present: true
}];

const arr = [
    {value: '9:55', source: 'camera', type: 'start'},
    {value: '10:55', source: 'computer', type: 'start'},
    {value: '12:50', source: 'camera', type: 'end'},
    {value: '13:10', source: 'camera', type: 'start'},
    {value: '15:10', source: 'computer', type: 'end'},
    {value: '15:30', source: 'camera', type: 'end'},
    {value: '15:50', source: 'camera', type: 'start'},
    {value: '16:05', source: 'computer', type: 'start'},
    {value: '20:00', source: 'camera', type: 'end'}
];

const getTime = (entry) => {
    const date = new Date();
    const [hours, min] = entry.split(':');
    date.setHours(hours, min, 0);
    return date.getTime();
};

function mergeAlgo(arr) {
    const combined = arr.map(it => ({...it, value: getTime(it.value)}));
    let isCamera = false;
    let isComputer = false;
    let start = null;
    const accum = [];

    combined.sort((a, b) => a.value - b.value);

    for (let i = 0; i < combined.length - 1; i++) {
        const current = combined[i];

        switch (current.source) {
            case 'computer':
                isComputer = !isComputer;
                break;
            case 'camera':
                isCamera = !isCamera;
                break;
        }

        if (isCamera && isComputer) {
            start = current.value;
        } else {
            if (start) {
                accum.push([start, current.value]);
                start = null;
            }
        }
    }

    return accum;
}

console.log(mergeAlgo(arr));

/**
 * @param {string} s
 * @return {boolean}
 */
var isNumber = function (s) {
    const n = s.trim();
    return /^[+-]?([0-9]+|[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)(e[+-]?[0-9]+)?$/i.test(n);
};

/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number}
 */
var minCameraCover = function (root) {
    var result = 0;
    /** tree travel function that return the state of a node */
    var travel = function (node) {
        if (!node) {
            return 2;
            //if return 0, we will need to add camera on leaf,
            //if return 1, means the leaf nodes will be monitored, and we will not add camera on leaf's parent node, so we have to return 2 here
        }
        var left = travel(node.left);
        var right = travel(node.right);
        // both childs are monitored
        if (left === 2 && right === 2) {
            return 0;
        }
        // one of the child is not monitored
        if (left === 0 || right === 0) {
            result++;
            return 1
        }
        // one of the child has camera
        if (left === 1 || right === 1) {
            return 2;
        }
        return; //I add a return here for my coding style, we have covered all the cases of childs, we must return either 0 or 1 or 2
    }
    //in case the root doesn't get monitored
    if (travel(root) === 0) result++;
    return result;
};

/**
 * @param {string[]} words
 * @return {number}
 */
var longestStrChain = function (words) {
    const memory = {};
    words.sort((a, b) => a.length - b.length);

    for (const word of words) {
        let longest = 0;
        for (let i = 0; i < word.length; i++) {
            const pre = word.slice(0, i) + word.slice(i + 1);
            longest = Math.max(longest, (memory[pre] || 0) + 1);
        }
        memory[word] = longest
    }

    return Math.max(...Object.values(memory));
};

/**
 * @param {string[]} paths
 * @return {string[][]}
 */
const findDuplicate = (paths) => {
    let m = new Map();
    for (const p of paths) {
        let pa = p.split(" ");
        for (let i = 1; i < pa.length; i++) {
            let start = pa[i].indexOf('(');
            let fileName = pa[i].slice(0, start);
            let content = pa[i].slice(start + 1, -1);
            if (!m.has(content)) m.set(content, []);
            m.get(content).push(pa[0] + '/' + fileName);
        }
    }
    return Array.from(m.values()).filter(x => x.length > 1);
};

/**
 * @param {number[]} nums
 * @return {number}
 */
var minMoves2 = function (nums) {
    const mid = nums.sort((a, b) => a - b)[~~(nums.length / 2)]
    return nums.reduce((a, c) => a + Math.abs(mid - c), 0)
};

/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[][]}
 */
var levelOrder = function (root) {
    if (!root) {
        return [];
    }

    const stack = [[root, 0]];
    const map = {};

    while (stack.length) {
        let [node, level] = stack.pop();

        if (map[level]) {
            map[level].push(node.val);
        } else {
            map[level] = [node.val];
        }

        if (node.right) {
            stack.push([node.right, level + 1])
        }

        if (node.left) {
            stack.push([node.left, level + 1])
        }
    }

    return Object.values(map);
};

/**
 * @param {string[]} words
 * @param {string} pattern
 * @return {string[]}
 */
var findAndReplacePattern = function (words, pattern) {
    function encode(str) {
        const result = [];
        const map = {};
        const source = str.split('');
        let cnt = 0;
        for (let symb of source) {
            if (map[symb]) {
                cnt++;
            }
            map[symb] = symb;
            if (source.indexOf(symb) === source.lastIndexOf(symb)) {
                result.push(`u${cnt}`);
            } else {
                result.push(`n${cnt}`);
            }
        }
        return result.join('');
    }

    const referrer = encode(pattern);
    return words.filter(word => encode(word) === referrer);
};

/**
 * Initialize your data structure here.
 * @param {number} n
 */
var TicTacToe = function (n) {
    this.row = new Array(n).fill(0);
    this.col = new Array(n).fill(0);
    this.diagonal = new Array(2).fill(0);
    this.size = n;
};

/**
 * Player {player} makes a move at ({row}, {col}).
 @param row The row of the board.
 @param col The column of the board.
 @param player The player, can be either 1 or 2.
 @return The current winning condition, can be either:
 0: No one wins.
 1: Player 1 wins.
 2: Player 2 wins.
 * @param {number} row
 * @param {number} col
 * @param {number} player
 * @return {number}
 */
TicTacToe.prototype.move = function (row, col, player) {
    let update = player === 1 ? 1 : -1;
    this.row[row] += update
    this.col[col] += update
    if (row == col) this.diagonal[0] += update;
    if (row == this.size - col - 1) this.diagonal[1] += update;
    if (Math.abs(this.row[row]) == this.size || Math.abs(this.col[col]) == this.size || Math.abs(this.diagonal[0]) == this.size || Math.abs(this.diagonal[1]) == this.size) {
        return player;
    }
    return 0;
};

/**
 * Your TicTacToe object will be instantiated and called as such:
 * var obj = new TicTacToe(n)
 * var param_1 = obj.move(row,col,player)
 */

/**
 * @param {string[]} words
 * @return {string}
 */
const shortestSuperstring = (A) => {
    function additionalStr(a, b) { // a + b additional Str
        for (let i = 0; i < a.length; i++) {
            if (b.startsWith(a.slice(i))) {
                return b.slice(a.length - i);
            }
        }
        return b;
    }

    // dp[i][j] -> min str building j state ending word i
    let dp = new Array(A.length).fill()
        .map(() => new Array(1 << A.length).fill(A.join('')));

    for (let s = 1; s < (1 << A.length); s++) {
        for (let j = 0; j < A.length; j++) {
            if (!(s & (1 << j))) continue;
            if (s === (1 << j)) dp[j][s] = A[j];
            let prevS = s ^ (1 << j);
            for (let k = 0; k < A.length; k++) {
                if (prevS & (1 << k)) {
                    let curStr = dp[k][prevS];
                    let tempStr = curStr + additionalStr(curStr, A[j]);
                    if (tempStr.length < dp[j][s].length) {
                        dp[j][s] = tempStr;
                    }
                }
            }
        }
    }
    let min = Number.MAX_VALUE;
    let minStr = null;
    for (let i = 0; i < A.length; i++) {
        let cur = dp[i][(1 << (A.length)) - 1];
        if (cur.length < min) {
            min = cur.length;
            minStr = cur;
        }
    }
    return minStr;
}

/**
 * @param {string} str
 * @return {string}
 */
var toLowerCase = function (str) {
    return str.toLowerCase()
};

// JS
const evaluate = {
    '+': (a, b) => a + b,
    '-': (a, b) => a - b,
    '*': (a, b) => a * b,
    '/': (a, b) => Math.trunc(a / b),
};

var evalRPN = function (tokens) {
    const stack = [];

    while (tokens.length) {
        const t = tokens.shift();
        if (t in evaluate) {
            const b = stack.pop();
            const a = stack.pop();
            stack.push(evaluate[t](a, b));
        } else stack.push(Math.trunc(Number(t)));
    }

    return stack[0];
};

/**
 * @param {string} n
 * @return {number}
 */
const minPartitions = n => Math.max(...n);

/**
 * @param {string[]} words
 * @return {number}
 */
var maxProduct = function (words) {
    const lengths = words.map(word => word.length);
    let max = 0;

    for (let i = 0; i < words.length; i++) {
        for (let j = 0; j < words.length; j++) {
            if (i == j) {
                continue;
            }

            let current = words[i].length * words[j].length;
            if (current > max && noCommonLetters(words[i], words[j])) {
                max = current;
            }
        }
    }

    function noCommonLetters(first, second) {
        for (let i = 0; i < first.length; i++) {
            if (second.includes(first[i])) {
                return false;
            }
        }

        return true;
    }

    return max;
};

/**
 * @param {number[]} nums
 * @return {number}
 */
var maximumUniqueSubarray = function (nums) {
    let start = 0;
    let end = 0;
    let current_sum = 0;
    let max_sum = 0;
    let set = new Set();
    while (end < nums.length) {
        if (!set.has(nums[end])) {
            set.add(nums[end]);
            current_sum += nums[end];
            max_sum = Math.max(max_sum, current_sum);
            end++
        } else {
            let deleteEl = nums[start];
            set.delete(deleteEl);
            current_sum = current_sum - deleteEl;
            start++;
        }
    }
    return max_sum
};

const connectSticks = sticks => {
    if (sticks.length <= 1) return 0
    // sorting at the beginning
    sticks.sort((a, b) => a - b)
    const combined = []
    let res = 0
    while (sticks.length || combined.length > 1) {
        let curSum = 0
        // we need to sticks to combine => counter is 2
        let counter = 2
        while (counter--) {
            // if we have original stick and it is less than first combined -> take original
            const condition = sticks.length && (!combined.length || (sticks[0] < combined[0]))
            // add to curSum and remove first from either combined or original sticks array
            curSum += condition ? sticks.shift() : combined.shift()
        }
        // add to result and add the stick that we combined to combined array
        res += curSum
        combined.push(curSum)
    }
    return res
};

/**
 * @param {number[]} nums
 * @return {number}
 */
var maximumGap = function (nums) {
    if (nums.length < 2) return 0;
    nums = nums.sort((a, b) => a - b);
    let max = 0;
    for (let i = 0; i < nums.length - 1; i++) {
        if (nums[i + 1] - nums[i] > max)
            max = nums[i + 1] - nums[i];
    }
    return max;
};

/**
 * @param {string[]} products
 * @param {string} searchWord
 * @return {string[][]}
 */
var suggestedProducts = function (products, searchWord) {
    products.sort();
    const result = [];
    let str = "";

    for (let letter of searchWord) {
        str += letter;
        result.push(products.filter(item => new RegExp(`^${str}`).test(item)).slice(0, 3));
    }

    return result;
};

/**
 * @param {number[][]} costs
 * @return {number}
 */
var minCost = function (costs) {

    let red = 0;
    let green = 1;
    let blue = 2;

    for (let i = 1; i < costs.length; i++) {

        costs[i][red] = costs[i][red] + Math.min(costs[i - 1][blue], costs[i - 1][green]);
        costs[i][blue] = costs[i][blue] + Math.min(costs[i - 1][red], costs[i - 1][green]);
        costs[i][green] = costs[i][green] + Math.min(costs[i - 1][blue], costs[i - 1][red]);

    }

    return Math.min(...costs[costs.length - 1]);
};

/**
 * @param {string} s1
 * @param {string} s2
 * @param {string} s3
 * @return {boolean}
 */
var isInterleave = function (s1, s2, s3) {
    if (s1.length + s2.length !== s3.length) return false;
    const memo = new Map()

    function run(i1, i2, i3) {
        if (i1 === s1.length) return s3.slice(i3) === s2.slice(i2);
        if (i2 === s2.length) return s3.slice(i3) === s1.slice(i1);
        const key = `${i1}-${i2}`;
        if (memo.has(key)) return memo.get(key)

        let res = false;

        if (s1[i1] === s2[i2] && s1[i1] === s3[i3]) res = run(i1 + 1, i2, i3 + 1) || run(i1, i2 + 1, i3 + 1);
        else if (s1[i1] === s3[i3]) res = run(i1 + 1, i2, i3 + 1);
        else if (s2[i2] === s3[i3]) res = run(i1, i2 + 1, i3 + 1);

        memo.set(key, res);
        return res;
    }

    return run(0, 0, 0)
};

/**
 * @param {number} h
 * @param {number} w
 * @param {number[]} horizontalCuts
 * @param {number[]} verticalCuts
 * @return {number}
 */
var maxArea = function (h, w, horizontalCuts, verticalCuts) {
    let M = Math.pow(10, 9) + 7;
    verticalCuts.push(w);
    horizontalCuts.push(h);
    verticalCuts.sort((a, b) => a - b);
    horizontalCuts.sort((a, b) => a - b);
    let maxWidth = 0;
    let maxHeight = 0;
    for (let i = 0; i < verticalCuts.length; i++) {
        let newWidth = i > 0 ? verticalCuts[i] - verticalCuts[i - 1] : verticalCuts[i]
        maxWidth = Math.max(maxWidth, newWidth);
    }
    for (let i = 0; i < horizontalCuts.length; i++) {
        let newHeight = i > 0 ? horizontalCuts[i] - horizontalCuts[i - 1] : horizontalCuts[i];
        maxHeight = Math.max(maxHeight, newHeight);
    }
    return (maxWidth * maxHeight) % M;
    // T.C: O(Mlog(M) + Nlog(N)), M = length of horizontalCuts and N = length of verticalCuts
    // S.C: O(1)
};

/**
 * @param {string[]} deadends
 * @param {string} target
 * @return {number}
 */
var openLock = function (deadends, target) {
    if (target === "0000") return 0
    let queue = [0], seen = new Uint8Array(10000)
    for (let d of deadends)
        seen[~~d] = 1
    target = ~~target
    if (seen[0]) return -1
    for (let turns = 1; queue.length; turns++) {
        let qlen = queue.length
        for (let i = 0; i < qlen; i++) {
            let curr = queue.shift()
            for (let j = 1; j < 10000; j *= 10) {
                let mask = ~~(curr % (j * 10) / j),
                    masked = curr - (mask * j)
                for (let k = 1; k < 10; k += 8) {
                    let next = masked + (mask + k) % 10 * j
                    if (seen[next]) continue
                    if (next === target) return turns
                    seen[next] = 1
                    queue.push(next)
                }
            }
        }
    }
    return -1
};

/**
 * @param {number} n
 * @param {number[]} speed
 * @param {number[]} efficiency
 * @param {number} k
 * @return {number}
 */
var maxPerformance = function (n, speeds, efficiencies, k) {
    const workers = new Array(n);
    for (let i = 0; i < n; i++) {
        workers[i] = {speed: BigInt(speeds[i]), efficiency: efficiencies[i]}
    }
    workers.sort((a, b) => b.efficiency - a.efficiency)

    let totalSpeed = BigInt(0);
    let max = BigInt(0);
    const heap = new MinHeap();
    for (let worker of workers) {
        const {speed, efficiency} = worker;
        totalSpeed += speed;

        heap.push(worker);
        if (heap.size() > k) totalSpeed -= heap.pop().speed;
        const total = totalSpeed * BigInt(efficiency)
        if (total > max) max = total;
    }
    return max % BigInt(1000000007);
};

class MinHeap {
    constructor() {
        this.store = [];
    }

    size() {
        return this.store.length;
    }

    isEmpty() {
        return this.store.length === 0;
    }

    push(value) {
        this.store.push(value);
        this.heapifyUp(this.store.length - 1);
    }

    pop() {
        if (this.store.length < 2) return this.store.pop();
        const result = this.store[0];
        this.store[0] = this.store.pop();
        this.heapifyDown(0);
        return result;
    }

    heapifyDown(parent) {
        while (true) {
            let [child, child2] = [1, 2].map((n) => parent * 2 + n).filter((n) => n < this.store.length);
            if (this.shouldSwap(child2, child)) {
                child = child2;
            }
            if (this.shouldSwap(child, parent)) {
                [this.store[child], this.store[parent]] = [this.store[parent], this.store[child]]
                parent = child;
            } else {
                return parent;
            }
        }
    }

    heapifyUp(child) {
        while (child) {
            const parent = Math.floor((child - 1) / 2);
            if (this.shouldSwap(child, parent)) {
                [this.store[child], this.store[parent]] = [this.store[parent], this.store[child]]
                child = parent;
            } else {
                return child;
            }
        }
    }

    shouldSwap(child, parent) {
        return child && this.store[child].speed < this.store[parent].speed
    }
}

/**
 * @param {number[]} nums
 * @return {number}
 */
var longestConsecutive = function (nums) {
    const map = new Map();
    let i = 0,
        maxLen = 0,
        len = 1;

    while (i < nums.length) {
        map.set(nums[i], 0);
        ++i;
    }
    map.forEach((value, key) => {
        if (value === 0) {
            map.set(key, 1);
            len = 1;

            while (map.has(key + len)) {
                map.set(key + len, 1);
                ++len;
            }
            maxLen = Math.max(maxLen, len);
        }
    });
    return maxLen;
}

/**
 * @param {number[]} cost
 * @return {number}
 */
var minCostClimbingStairs = function (cost) {
    /*
    Let dp[i] be the cost of reaching at step 'i', its not the cost we pay on 'i' its just the price we pay to reach till this step.
    */
    let dp = [];
    //We know we can reach in 0 cost on step 0 and 0+1
    dp[0] = 0;
    dp[1] = 0;
    for (let i = 2; i <= cost.length; i++) {
        //For any other step, we can either come from just previous step (i-1) or from previous to previous step (i-2). In both the cases we will consider the cost we paid of reaching to that step (dp[i-1] and dp[i-2]) and the cost we paid on that step (cost[i-1] and cos[i-2]) to jump on  'i'
        dp[i] = Math.min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
    }


    return dp[cost.length];
};

const helper = (preorder, map, p1, p2, i1, i2) => {
    if (p1 >= p2 || i1 >= i2) return null;
    const inorderIdx = map.get(preorder[p1]); // Inorder root index
    const leftSubtreeSize = inorderIdx - i1;
    return new TreeNode(
        preorder[p1],
        helper(preorder, map, p1 + 1, p1 + leftSubtreeSize + 1, i1, inorderIdx),
        helper(preorder, map, p1 + leftSubtreeSize + 1, p2, inorderIdx + 1, i2)
    );
};

const buildTree = (preorder, inorder) => {
    const map = new Map();
    for (let i = 0; i < inorder.length; i++) map.set(inorder[i], i);
    return helper(preorder, map, 0, preorder.length, 0, inorder.length);
};

/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number}
 */
var maxResult = function (nums, k) {
    let n = nums.length, deq = [n - 1]
    for (let i = n - 2; ~i; i--) {
        if (deq[0] - i > k) deq.shift()
        nums[i] += nums[deq[0]]
        while (deq.length && nums[deq[deq.length - 1]] <= nums[i]) deq.pop()
        deq.push(i)
    }
    return nums[0]
};

var MyCalendar = function () {
    this.cal = []
};

MyCalendar.prototype.book = function (start, end) {
    let l = 0, r = this.cal.length - 1
    while (l <= r) {
        const mid = Math.floor((r + l) / 2)
        const [s, e] = this.cal[mid]
        if (s < end && start < e) return false
        if (start >= e) {
            l = mid + 1
        } else {
            r = mid - 1
        }
    }
    this.cal.splice(l, 0, [start, end])
    return true
};

/**
 * Your MyCalendar object will be instantiated and called as such:
 * var obj = new MyCalendar()
 * var param_1 = obj.book(start,end)
 */

/**
 * @param {number[]} stones
 * @return {number}
 */
var stoneGameVII = function (s) {
    let len = s.length, dp = new Array(len).fill().map(_ => new Array(len).fill(0))
    for (let i = len - 2; ~i; i--)
        for (let j = i + 1, sum = s[i] + s[j]; j < len; sum += s[++j])
            dp[i][j] = Math.max(sum - s[i] - dp[i + 1][j], sum - s[j] - dp[i][j - 1])
    return dp[0][len - 1]
};

/**
 * @param {number} target
 * @param {number} startFuel
 * @param {number[][]} stations
 * @return {number}
 */
var minRefuelStops = function (target, startFuel, stations) {
    let curr = startFuel;
    let len = stations.length;
    stations.sort(function (x, y) {
        return x[0] - y[0]
    });
    let pq = new Array();
    let i = 0, count = 0;
    while (curr < target) {
        count++;
        while (i < len && stations[i][0] <= curr) {
            pq.push(stations[i][1]);
            i++;
        }
        if (!pq.length)
            break;
        let max = Math.max(...pq);
        curr += max;
        pq.splice(pq.indexOf(max), 1);
    }
    return curr >= target ? count : -1;
};

var palindromePairs = function (words) {
    var i, j, wordLength, prefix, suffix, reversedPrefix, reversedSuffix;
    var dict = {};
    var result = [];
    var length = words.length;

    if (!words || length === 0) {
        return [];
    }

    for (i = 0; i < length; i += 1) {
        dict[words[i]] = i;
    }

    for (i = 0; i < length; i += 1) {
        wordLength = words[i].length;

        prefix = '';
        suffix = words[i];
        reversedPrefix = '';
        reversedSuffix = suffix.split('').reverse().join('');

        for (j = 0; j < wordLength + 1; j += 1) {
            if (j !== 0) {
                prefix += words[i][j - 1];
                suffix = suffix.slice(1);
                reversedPrefix = words[i][j - 1] + reversedPrefix;
                reversedSuffix = reversedSuffix.slice(0, reversedSuffix.length - 1);
            }

            if (j !== 0 && prefix === reversedPrefix && reversedSuffix in dict && dict[reversedSuffix] !== i) {
                result.push([dict[reversedSuffix], i]);
            }

            if (suffix === reversedSuffix && reversedPrefix in dict && dict[reversedPrefix] !== i) {
                result.push([i, dict[reversedPrefix]]);
            }
        }
    }
    return result;
};

var maximumUnits = function (boxTypes, truckSize) {
    boxTypes.sort((a, b) => b[1] - a[1]);

    let units = 0;

    for (const [boxes, boxUnits] of boxTypes) {
        if (truckSize === 0) return units;
        const boxesToPlace = Math.min(truckSize, boxes);
        truckSize -= boxesToPlace;
        units += (boxesToPlace * boxUnits);
    }

    return units;
};

var makesquare = function (nums) {
    if (nums == null || nums.length == 0) return false;
    let sum = 0;
    for (let num of nums) sum += num;
    if (sum % 4 != 0) return false;
    nums.sort((a, b) => a - b);
    return dfs(nums, new Array(4).fill(0), nums.length - 1, sum / 4);
};
var dfs = function (nums, sums, index, target) {
    if (index == -1) return true;
    for (let i = 0; i < 4; i++) {
        if (sums[i] + nums[index] > target || (i > 0 && sums[i] == sums[i - 1])) continue;
        sums[i] += nums[index];
        if (dfs(nums, sums, index - 1, target)) return true;
        sums[i] -= nums[index];
    }
    return false;
};

/**
 * @param {number} n
 * @return {string[]}
 */
var generateParenthesis = function (n) {
    const results = [];

    function helper(left, right, max, str) {
        if (left === max && right === max) {
            results.push(str);
            return;
        }

        if (left < max) {
            helper(left + 1, right, max, `${str}(`)
        }

        if (right < left) {
            helper(left, right + 1, max, `${str})`)
        }
    }

    helper(0, 0, n, '')

    return results;
};

/**
 * @param {number[]} nums
 * @param {number} left
 * @param {number} right
 * @return {number}
 */
var numSubarrayBoundedMax = function (nums, left, right) {

    let lastInRange = -1, lastOverBound = -1, sum = 0;

    for (let i = 0; i < nums.length; i++) {

        const el = nums[i];

        if (el > right) {
            lastOverBound = i;
            continue;
        }

        if (el >= left && el <= right) {
            lastInRange = i;
        }

        if (lastInRange <= lastOverBound) continue;

        sum += lastInRange - lastOverBound;
    }

    return sum;
};

class NumArray {
    constructor(nums) {
        this.nums = new Int8Array(nums);
        this.bit = new Int32Array(nums.length + 1).fill(0);

        for (let i = 0; i < nums.length; ++i) {
            this.insert(i + 1, nums[i]);
        }
    }

    insert(i, diff) {
        while (i < this.bit.length) {
            this.bit[i] += diff;
            i += i & -i;
        }
    }

    update(i, val) {
        this.insert(i + 1, val - this.nums[i]);
        this.nums[i] = val;
    }

    runningSum(i) {
        let sum = 0;

        while (i > 0) {
            sum += this.bit[i];
            i -= (i & -i);
        }

        return sum;
    }

    sumRange(leftIdx, rightIdx) {
        return this.runningSum(rightIdx + 1) - this.runningSum(leftIdx);
    }
}

/**
 * @param {number} n
 * @param {number} k
 * @return {number}
 */
var kInversePairs = function (n, k) {
    let mod = 1000000007;
    if (k > n * (n - 1) / 2 || k < 0) return 0;
    let dp = new Array(n + 1).fill(0).map(() => new Array(k + 1).fill(0));
    for (let i = 1; i <= n; i++) {
        dp[i][0] = 1;
        if (i + 1 <= n) dp[i + 1][0] = 1;
        for (let j = 1; j <= Math.min(k, i * (i - 1) / 2); j++) {
            dp[i][j] = dp[i][j - 1] + dp[i - 1][j];
            if (j >= i) dp[i][j] -= dp[i - 1][j - i];
            dp[i][j] = (dp[i][j] + mod) % mod;
        }
    }
    return dp[n][k];
};

var swimInWater = function (grid) {
    let visited = new Set(), time = 0, N = grid.length;
    let dirs = [[-1, 0], [0, -1], [0, 1], [1, 0]];

    const dfs = (r, c) => {
        if (r < 0 || r > N - 1 || c < 0 || c > N - 1 || time < grid[r][c] || visited.has(r * N + c)) return;
        visited.add(r * N + c);
        for (let [rr, cc] of dirs)
            dfs(r + rr, c + cc);
    };

    while (!visited.has(N * N - 1)) {
        visited.clear();
        dfs(0, 0);
        time++;
    }

    return time - 1;
};

/**
 * @param {number} numRows
 * @return {number[][]}
 */
var generate = function (numRows) {
    const rows = [[1], [1, 1]];

    if (!numRows) {
        return []
    }

    if (numRows < 2) {
        return [rows[numRows - 1]];
    }

    for (let i = 2; i < numRows; i++) {
        const temp = [1];
        for (let j = 1; j < i; j++) {
            temp[j] = rows[i - 1][j - 1] + rows[i - 1][j];
        }
        temp.push(1);
        rows.push(temp);
    }

    return rows;
};

/**
 * @param {string} s
 * @param {string[]} words
 * @return {number}
 */
var numMatchingSubseq = function (s, words) {


    let alph = "abcdefghijklmnopqrstuvwxyz";
    let head = {};
    let count = 0;
    // create a hash of alph letter pointing to an empty array
    for (let char of alph) {
        head[char] = [];
    }
    // populate each array with the words that start with the corresponding letter
    for (let word of words) {
        head[word[0]].push(word);
    }

    let current;
    let temp = [];
    // iterate over s
    for (let char of s) {
        // white the array at the current char of s is not empty
        while (head[char].length) {
            //pop a value and slice the first letter out
            current = head[char].pop();
            current = current.slice(1);
            // if the string is equal to zero that means that all characters have been found
            if (!current.length) {
                count++;
            } else {
                //if not push to a temp array so that no word gets counted twice i.e. "bbb" becomes "bb" and then "b" and then "" and gets counted
                // as one in the same iteration because it is being pushed back to the array and the length never becomes zero
                temp.push(current);
            }
        }
        //push the temp array back to the hash
        for (t of temp) {
            head[t[0]].push(t);
        }
        // reset temp array
        temp = [];
    }
    // return the count
    return count;
};

/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} head
 * @param {number} left
 * @param {number} right
 * @return {ListNode}
 */
var reverseBetween = function (head, left, right) {
    if (!head || !head.next) return head;

    let dummy = new ListNode(-1, head);
    let prev = dummy,
        current = head,
        next = null;
    let z = 1;

    while (z < left && current) {
        prev = current;
        current = current.next;
        ++z;
    }
    while (z < right && current) {
        next = current.next.next;
        current.next.next = prev.next;
        prev.next = current.next;
        current.next = next;
        ++z;
    }
    return dummy.next;
};

var findPaths = function (m, n, N, i, j) {
    const dir = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    const memo = new Map();

    function callDFS(r, c, step) {
        if (r < 0 || c < 0 || r == m || c == n) return 1;
        if (!step) return 0;

        const key = `${r}-${c}-${step}`;
        if (memo.has(key)) return memo.get(key);

        let sum = 0;

        for (let [dr, dc] of dir) {
            sum += callDFS(r + dr, c + dc, step - 1);
            sum %= (10 ** 9 + 7);
        }
        memo.set(key, sum);
        return sum;
    }

    return callDFS(i, j, N);
};

var findRedundantConnection = function (edges) {
    let map = {};
    let currRedundant = null;

    const dfs = (u, v, visited) => {
        visited.add(u);

        if (u in map) {
            if (map[u].has(v)) return true;
            for (let w of map[u]) {
                if (!visited.has(w)) {
                    visited.add(w);
                    if (dfs(w, v, visited)) return true;
                }
            }
            return false;
        }
        return false;
    }

    for (let edge of edges) {
        let [u, v] = edge;

        if (dfs(u, v, new Set())) currRedundant = edge;

        if (!(u in map)) map[u] = new Set();
        if (!(v in map)) map[v] = new Set();

        map[u].add(v);
        map[v].add(u);
    }

    return currRedundant;
};

var countSmaller = function (nums) {
    if (nums.length === 0) return [];

    let counts = [0];
    let orderedNums = [nums[nums.length - 1]];

    const bs = (target) => {
        let start = 0;
        let end = orderedNums.length;
        while (start < end) {
            let mid = Math.floor((start + end) / 2);
            if (target > orderedNums[mid]) start = mid + 1;
            else end = mid;
        }
        orderedNums.splice(start, 0, target);
        return start;
    }

    for (let i = nums.length - 2; i >= 0; i--) {
        let count = bs(nums[i]);
        counts.unshift(count);
    }
    return counts;
};

/**
 * @param {number[]} ratings
 * @return {number}
 */
var candy = function (ratings) {
    var len = ratings.length;
    var forward = new Array(len);
    let backward = new Array(len);

    forward[0] = 1;
    backward[len - 1] = 1;
    for (var i = 1; i < len; i++) {
        if (ratings[i] > ratings[i - 1]) forward[i] = forward[i - 1] + 1;
        else forward[i] = 1;
    }

    for (var i = len - 2; i >= 0; i--) {
        if (ratings[i] > ratings[i + 1]) backward[i] = backward[i + 1] + 1;
        else backward[i] = 1;
    }
    var sum = 0;
    for (var i = 0; i < len; i++) {
        sum += Math.max(forward[i], backward[i]);
    }
    return sum;
};

/**
 * @param {string} S
 * @return {string}
 */
var removeDuplicatesTwo = function (S) {
    let indexes = [];
    for (let i = 0; i < S.length; i++) {

        if (S[i] === S[i + 1]) {
            indexes.push(i);
            indexes.push(i + 1);
            return removeDuplicates(S.replace(S.substr(indexes[0], indexes.length), ''));
        }
    }

    return S;
};

/**
 * @param {number} n
 * @return {boolean}
 */
var isArmstrong = function (n) {
    let v = n;
    let c, tot = 0;
    const ln = n.toString().length;

    while (v > 0) {
        c = v % 10;
        v = Math.floor(v / 10);
        tot += c ** ln
    }

    return tot === n;
};

/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @param {TreeNode} p
 * @param {TreeNode} q
 * @return {TreeNode}
 */
var lowestCommonAncestor = function (root, p, q) {
    if (!root || root === p || root === q) return root;
    var left = lowestCommonAncestor(root.left, p, q);
    var right = lowestCommonAncestor(root.right, p, q);
    if (left && right) return root;
    if (!left) return right;
    return left;
};

/**
 * @param {number} n
 * @return {number[]}
 */
var grayCode = function (n) {
    let list = []
    let stack = [];
    list.push(0);
    let i = 0;
    while (i < n) {
        let len = list.length;
        let inc = (Math.pow(2, i));
        for (let j = 0; j < len; j++) {
            stack.push(list[j]);
        }
        while (stack.length > 0) {
            list.push(stack.pop() + inc);
        }
        i++;
    }
    return list;
};

/**
 * @param {number[]} arr
 * @param {number} k
 * @param {number} x
 * @return {number[]}
 */
var findClosestElements = function (arr, k, x) {
    let closestChain = [];
    for (let i = 0; i < arr.length; i++) {
        let mapIn = {};
        mapIn.val = arr[i];
        mapIn.subs = Math.abs(arr[i] - x);
        closestChain.push(mapIn);
    }
    closestChain.sort((a, b) => a.subs - b.subs);
    return closestChain.slice(0, k).map(a => a.val).sort((a, b) => a - b);
};

/**
 * @param {number[][]} matrix
 * @param {number} k
 * @return {number}
 */
var maxSumSubmatrix = function (m, k) {
    let max = -Infinity;
    const rows = m.length, cols = m[0].length;
    const sums = Array(rows);
    for (let c1 = 0; c1 < cols; c1++) {
        sums.fill(0);
        for (let c2 = c1; c2 < cols; c2++) {
            for (let row = 0; row < rows; row++) {
                sums[row] += m[row][c2];
            }
            for (let i = 0; i < rows; i++) {
                let sum = 0;
                for (let j = i; j < rows; j++) {
                    sum += sums[j];
                    if (sum > max && sum <= k) {
                        max = sum;
                    }
                }
            }
        }
    }
    return max;
};

/**
 * @param {number} n
 * @return {number}
 */
var countVowelPermutation = function (n) {
    const MOD = 1000000007;
    const dp = Array(n + 1).fill(null).map(() => Array(5).fill(0));
    dp[1][0] = 1;    // a
    dp[1][1] = 1;    // e
    dp[1][2] = 1;    // i
    dp[1][3] = 1;    // o
    dp[1][4] = 1;    // u

    for (let i = 2; i <= n; ++i) {
        dp[i][0] = dp[i - 1][1];                                                       // ae
        dp[i][1] = (dp[i - 1][0] + dp[i - 1][2]) % MOD;                                // ea, ei
        dp[i][2] = (dp[i - 1][0] + dp[i - 1][1] + dp[i - 1][3] + dp[i - 1][4]) % MOD;  // ia, ie, io, iu
        dp[i][3] = (dp[i - 1][2] + dp[i - 1][4]) % MOD;                                // oi, ou
        dp[i][4] = dp[i - 1][0];                                                       // ua
    }
    return (dp[n][0] + dp[n][1] + dp[n][2] + dp[n][3] + dp[n][4]) % MOD;
};

/**
 * @param {number[][]} mat
 * @param {number} r
 * @param {number} c
 * @return {number[][]}
 */
var matrixReshape = function (mat, r, c) {
    const [m, n] = [mat.length, mat[0].length]

    if (m * n !== r * c) {
        return mat
    }

    const newMatrix = Array.from({length: r}, _ => Array(c))
    const reversedFlattenMatrix = mat.flat().reverse()

    for (let row = 0; row < r; row++) {
        for (let col = 0; col < c; col++) {
            newMatrix[row][col] = reversedFlattenMatrix.pop()
        }
    }

    return newMatrix
}

/**
 * @param {number[]} arr
 * @return {number}
 */
var minSetSize = function (arr) {
    const map = {};
    for (let elem of arr) {
        map[elem] = ++map[elem] || 1;
    }

    const counter = Object.entries(map).sort(([k1, v1], [k2, v2]) => v2 - v1);
    const N = arr.length / 2;
    let accum = 0;
    let toDelete = [];

    for (let [num, count] of counter) {
        if (accum < N) {
            toDelete.push(num);
            accum += +count;
        } else {
            break;
        }
    }

    return toDelete.length;
};

/**
 * @param {number[][]} matrix
 * @param {number} k
 * @return {number}
 */
var kthSmallest = function (matrix, k) {
    return [].concat(...matrix).sort((a, b) => a - b)[k - 1];
};

/**
 * @param {character[]} s
 * @return {void} Do not return anything, modify s in-place instead.
 */
var reverseWords = function (s) {
    s.reverse();
    let l = 0;
    for (let i = 0; i < s.length; i++) {
        if (s[i + 1] === ' ' || s[i + 1] === undefined) {
            let r = i;
            while (l < r) {
                let temp = s[l];
                s[l] = s[r];
                s[r] = temp;
                l++, r--;
            }
            l = i + 2;
        }
    }
}

/**
 * @param {number[]} nums
 * @return {number}
 */
var lengthOfLIS = function (nums) {
    var n = nums.length;
    var dp = new Array(n).fill(1);
    for (var i = 1; i < n; i++) {
        var max = 0;
        for (var j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                max = Math.max(max, dp[j]);
            }
        }
        dp[i] = Math.max(dp[i], max + 1);
    }
    return Math.max(...dp);
};

/**
 * @param {string} s
 * @return {number}
 */
var numDecodings = function (s) {
    let len = s.length;
    if (len === 0) return s;
    let mod = Math.pow(10, 9) + 7;
    let dp = new Array(len + 1).fill(0);
    dp[0] = 1;
    if (s[0] == 0) dp[1] = 0;
    else if (s[0] == '*') dp[1] = 9;
    else dp[1] = 1;
    for (let i = 2; i <= len; i++) {
        if (s[i - 1] == '*') {
            dp[i] += 9 * dp[i - 1];
        } else if (s[i - 1] != 0 && s[i - 1] != '*') {
            dp[i] += dp[i - 1];
        }
        if (s[i - 2] == 1) {
            if (s[i - 1] == '*') dp[i] += 9 * dp[i - 2];
            else dp[i] += dp[i - 2];
        } else if (s[i - 2] == 2) {
            if (s[i - 1] < 7) dp[i] += dp[i - 2];
            else if (s[i - 1] == '*') dp[i] += 6 * dp[i - 2];
        } else if (s[i - 2] == '*') {
            if (s[i - 1] == '*') dp[i] += 15 * dp[i - 2];
            else if (s[i - 1] >= 0 && s[i - 1] <= 6) dp[i] += 2 * dp[i - 2];
            else if (s[i - 1] >= 7) dp[i] += dp[i - 2];
        }
        dp[i] = dp[i] % mod;
    }
    return dp[len] % mod;
};

/**
 * initialize your data structure here.
 */
var MedianFinder = function () {
    this.small = new MaxPriorityQueue({priority: (num) => num});
    this.large = new MinPriorityQueue({priority: (num) => num});
};

/**
 * @param {number} num
 * @return {void}
 */
MedianFinder.prototype.addNum = function (num) {
    this.small.enqueue(num);
    this.large.enqueue(this.small.dequeue().element);
    if (this.small.size() < this.large.size()) {
        this.small.enqueue(this.large.dequeue().element);
    }
};

/**
 * @return {number}
 */
MedianFinder.prototype.findMedian = function () {
    return this.small.size() > this.large.size()
        ? this.small.front().element
        : (this.small.front().element + this.large.front().element) / 2;
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * var obj = new MedianFinder()
 * obj.addNum(num)
 * var param_2 = obj.findMedian()
 */

/**
 * @param {string} s
 * @param {string} t
 * @return {boolean}
 */
const isIsomorphic = function (s, t) {
    const hash1 = {}
    const hash2 = {}
    for (let idx = 0; idx < s.length; idx++) {
        if (hash1[s[idx]] !== hash2[t[idx]]) return false
        hash1[s[idx]] = idx
        hash2[t[idx]] = idx
    }
    return true
};

/**
 * @param {number[]} nums
 * @return {number}
 */
var findPeakElement = function (nums) {
    let [max, idx] = [-Infinity, -1]

    for (let i = 0; i < nums.length; i++) {
        if (nums[i] > max) {
            [max, idx] = [nums[i], i]
        }
    }

    return idx;
};

/**
 * @param {string} order
 * @param {string} str
 * @return {string}
 */
var customSortString = function (order, str) {
    const map = {};
    order.split('').map((c, i) => map[c] = i);
    return str.split('').sort((c1, c2) => (map[c1] ?? 30) - (map[c2] ?? 30)).join('');
};

var triangleNumber = function (nums) {
    nums.sort((a, b) => a - b);
    let counter = 0;

    for (let x = 0; x < nums.length - 2; x++) {
        for (let y = x + 1; y < nums.length - 1; y++) {
            let target = nums[x] + nums[y]
            let low = y, high = nums.length - 1;

            while (low < high) {
                const mid = Math.ceil((low + high) / 2);
                if (nums[mid] >= target) high = mid - 1;
                else low = mid;
            }
            counter += (low - y);
        }
    }
    return counter;
};

const fourSum = (nums, tar) => {
    nums.sort((a, b) => a - b);
    return kSum(nums, tar, 0, 4);
};

const kSum = (nums, tar, start, k) => {
    const res = [];
    if (start === nums.length || nums[start] * k > tar || tar > nums[nums.length - 1] * k) return res;
    if (k === 2) return twoSum(nums, tar, start);
    for (let i = start; i < nums.length; i++) {
        if (i === start || nums[i] !== nums[i - 1]) {
            for (let set of kSum(nums, tar - nums[i], i + 1, k - 1)) {
                set.push(nums[i]);
                res.push(set);
            }
        }
    }
    return res;
};

const twoSum = (nums, tar, start) => {
    const res = [];
    const set = new Set();
    for (let i = start; i < nums.length; i++) {
        if (!res.length || res[res.length - 1][1] !== nums[i])
            if (set.has(tar - nums[i])) res.push([tar - nums[i], nums[i]]);
        set.add(nums[i]);
    }
    return res;
};

var threeEqualParts = function (arr) {
    const ones = arr.reduce((s, n) => s + n, 0);
    if (ones === 0) return [0, 2];
    if (ones % 3 !== 0) return [-1, -1];
    let onesToFind = ones / 3;
    let k = arr.length;
    while (onesToFind > 0) if (arr[--k] === 1) --onesToFind;
    const iter = arr.length - k;
    const firstOne = arr.indexOf(1);
    const secondOne = arr.indexOf(1, firstOne + iter);
    for (let i = 0; i < iter; i++)
        if (arr[i + firstOne] !== arr[k + i] || arr[i + secondOne] !== arr[k + i]) return [-1, -1];
    return [firstOne + iter - 1, secondOne + iter];
};

/**
 * @param {number} length
 * @param {number[][]} updates
 * @return {number[]}
 */
var getModifiedArray = function (length, updates) {
    /*
    startIdxi, endIdxi, inci
    Approach: add 'inci' at the 'startIdxi' and substract 'inci' ast 'endIdxi'+1 if 'endIdxi'+1 inside the range
    Do sum of elements to create the 'ans' array
    */
    let ans = [], sum = 0;
    for (let i = 0; i < length; i++) {
        ans.push(0);
    }
    for (let i = 0; i < updates.length; i++) {
        ans[updates[i][0]] += updates[i][2];
        if (updates[i][1] + 1 <= length - 1) {
            ans[updates[i][1] + 1] -= updates[i][2];
        }
    }
    for (let i = 0; i < length; i++) {
        sum += ans[i];
        ans[i] = sum;
    }
    return ans;
};

/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */

/**
 * @param {TreeNode} root
 * @param {TreeNode} p
 * @param {TreeNode} q
 * @return {TreeNode}
 */
const lowestCommonAncestor = (root, p, q) => {
    let lca = root;
    while (lca) {
        if (p.val < lca.val && q.val < lca.val) lca = lca.left;
        else if (p.val > lca.val && q.val > lca.val) lca = lca.right;
        else return lca;
    }
    return null;
};

/**
 * @param {number[]} nums
 */
var Solution = function (nums) {
    this.nums = nums || []
};

/**
 * Resets the array to its original configuration and return it.
 * @return {number[]}
 */
Solution.prototype.reset = function () {
    return this.nums
};

/**
 * Returns a random shuffling of the array.
 * @return {number[]}
 */
Solution.prototype.shuffle = function () {
    // create a copy of this.nums, shuffle it, and return it
    const shuffled = this.nums.slice();
    const n = shuffled.length;
    const swap = (arr, i, j) => {
        let tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    // swap elements with random elements
    for (let i = 0; i < n; i++)
        swap(shuffled, i, Math.floor(Math.random() * n));

    return shuffled;
};

/**
 * Your Solution object will be instantiated and called as such:
 * var obj = new Solution(nums)
 * var param_1 = obj.reset()
 * var param_2 = obj.shuffle()
 */

/**
 * @param {string} dominoes
 * @return {string}
 */
var pushDominoes = function (dominoes) {
    let l = 0, r = 1;
    const arr = ("L" + dominoes + "R").split("");
    while (l < arr.length - 1) {
        while (arr[r] == '.')
            r++;
        if (arr[l] == arr[r])
            for (let i = l + 1; i < r; i++)
                arr[i] = arr[l];
        if (arr[l] > arr[r])
            for (let i = 1; i <= (r - l - 1) / 2; i++) {
                arr[l + i] = 'R';
                arr[r - i] = 'L';
            }
        l = r++;
    }
    return arr.slice(1, arr.length - 1).join("");
};

/**
 * @param {number[]} nums
 * @return {number}
 */
var partitionDisjoint = function (A) {
    let maxLeft = A[0];
    let maxSoFar = A[0]
    let pivot = 0;
    for (let i = 1; i < A.length; i++) {
        if (A[i] > maxSoFar) {
            maxSoFar = A[i];
        }
        if (A[i] < maxLeft) {
            maxLeft = maxSoFar;
            pivot = i;
        }
    }
    return pivot + 1;
};

/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {TreeNode}
 */
var pruneTree = function (root) {
    //pruning left and right, if they are available
    //if you have base case check as in -> if(!root) return null, then no need of the && condition
    root.left = root.left && pruneTree(root.left);
    root.right = root.right && pruneTree(root.right);

    //if all conditions are met, prune(return null) else keep the node
    return (root.val === 0 && !root.left && !root.right) ? null : root;
};

/**
 * @param {string[]} words
 * @return {string}
 */
function alienOrder(words) {
    let uniqueCharCount = new Set(), indegree = {}, dependencyArray = [], indegreeQueue = [], uniqueCharString = "";

    //Collect all unique characters and initialize indegree object
    for (let word of words) {
        for (let char of word) {
            if (!uniqueCharCount.has(char)) {
                uniqueCharCount.add(char);
                indegree[char] = 0;
            }
        }
    }
    //Populate indegree object with incoming edges
    for (let i = 0; i < words.length - 1; i++) {
        if (words[i].startsWith(words[i + 1]) && words[i].length > words[i + 1].length) {
            return "";
        }
        let minWordLength = Math.min(words[i].length, words[i + 1].length), differenceFound = false;
        for (let j = 0; j < minWordLength && !differenceFound; j++) {
            if (words[i][j] !== words[i + 1][j]) {
                dependencyArray.push([words[i + 1][j], words[i][j]]);
                indegree[words[i + 1][j]]++;
                differenceFound = true;
            }
        }
    }
    //Build return string with characters that have 0 incoming edges
    for (let key in indegree) {
        if (indegree[key] === 0) {
            indegreeQueue.push(key);
            uniqueCharString += key;
        }
    }
    //Conduct BFS to remove incoming edges for each character and append characters with 0 incoming edges to the return string
    while (indegreeQueue.length) {
        let currentNode = indegreeQueue.shift();
        for (let i = 0; i < dependencyArray.length; i++) {
            if (dependencyArray[i][1] === currentNode) {
                indegree[dependencyArray[i][0]]--;
                if (indegree[dependencyArray[i][0]] === 0) {
                    indegreeQueue.push(dependencyArray[i][0]);
                    uniqueCharString += dependencyArray[i][0];
                }
            }
        }
    }
    //Return the return string if the length is the same as the count of unique characters, else return empty string
    return uniqueCharString.length === uniqueCharCount.size ? uniqueCharString : "";
};

/**
 * @param {number} n
 * @return {number}
 */
var findIntegers = function (n) {
    let dp = len => {
        if (len < 0)
            return 0;
        if (!len)
            return 1;
        let _0x = 1; // number of accepted combination when '1' is first
        let _1x = 1; // number of accepted combination when '0' is first
        while (--len)
            [_0x, _1x] = [_0x + _1x, _0x];
        return _0x + _1x;
    };
    let binary = n.toString(2);
    let count = 0;
    let is_prev_one = false;
    for (let i = 0; i < binary.length; i++) {
        if (binary[i] === '0') {
            is_prev_one = false;
            continue;
        }
        count += dp(binary.length - i - 1);
        if (is_prev_one)
            return count;
        is_prev_one = true;
    }
    return count + 1;
};

