import java.util.Arrays;
import java.util.HashMap;
import java.util.Stack;


class Solution {
    public class ListNode {
       int val;
        ListNode next;
        ListNode(int x) { val = x; } 
    }
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }
    //1108
    public static String defangIPaddr(String address) {
        return address.replace(".", "[.]");
    }
    //1281
    public int subtractProductAndSum(int n) {
        int product = 1;
        int sum = 0;
        while(n>0) {
            product *= (n%10);
            sum += (n%10);
            n/=10;
        }
        return (product - sum);
    }
    //1295
    public static int findNumbers(int[] nums) {
        int result = 0 ;
        for (int i =0; i< nums.length; i++){
            String s = Integer.toString(nums[i]);
            if(s.length() % 2 == 0) {
                result++;
            }
        }
        return result;
    }
    //771
    public int numJewelsInStones(String J, String S) {
        char[] jewels = J.toCharArray();
        char[] stones = S.toCharArray();
        int found = 0;
        l1: for(char s : stones) {
            for(char j : jewels) {
                if(s == j) {
                    found++;
                    continue l1;
                }
            }
        }
        
        return found; 
    }
    //1290
    public int getDecimalValue(ListNode head) {
        String  combString = "";
        while (head != null) {
            combString += Integer.toString(head.val);
            head = head.next;
        }
        int result = 0;
        int bit = combString.length();
        for (int i =0 ; i< combString.length(); i++) {
            if (combString.charAt(i) == '1') {
                result += Math.pow(2,bit-i-1);
            }
        }
        return result;
    }
    //1221
    public int balancedStringSplit(String s) {
        Stack<Character> stack = new Stack<>();
        int result = 0;
        for (int i =0 ; i< s.length(); i++) {
            if (!stack.isEmpty()){
                if(s.charAt(i) == 'L' && stack.peek() == 'R'){
                    stack.pop();
                    if (stack.isEmpty()){
                        result ++;
                    }
                }else if(s.charAt(i) == 'R' && stack.peek() == 'L'){
                    stack.pop();
                    if (stack.isEmpty()){
                        result ++;
                    } 
                } else {
                    stack.push(s.charAt(i));
                }
            }else {
                stack.push(s.charAt(i));
            }
        }
        return result;
    }
    public int balancedStringSplitAnotherFasterSolution(String s) {
        int r=0,l=0;
        int res=0;
        for(int i=0; i<s.length(); i++){
            if(s.charAt(i)=='L'){l++;}
            else{r++;}
            if(r==l){
                l=0;
                r=0;
                res++;
            }
        }
        return res;
    }
    // 938
    int result = 0;
    public int rangeSumBST(TreeNode root, int L, int R) {
        if(root != null) {
            rangeSumBST(root.left, L, R);
            if (root.val <= R && root.val >= L) {
                result += root.val;
            }
            rangeSumBST(root.right, L, R);
        }
        return result;

    }
    //1021
    public String removeOuterParentheses(String S) {
        int count = 0; // count the order of the parantheses
        int check = 0; // check the validity of the parantheses
        String result  = "";
        for (int  i = 0 ; i < S.length(); i++){
            if (check == 0 && S.charAt(i) == '('){
                check ++;
                count ++;
            } else if (check != 0 && S.charAt(i) == '(' ){
                count++;
                check++;
                result += '(';
            } else if (check != 1 && S.charAt(i) == ')' && count != 1) {
                check --;
                result += ')';
            } else {
                check --;
                count --;
            }
        }
        return result;
    }

    //1051
    public int heightChecker(int[] heights) {
        int count = 0;
        int[] newA = heights.clone();
        Arrays.sort(newA);
        for (int i =0; i< newA.length;i++) {
            if (heights[i] != newA[i]) {
                count++;
            }
        }
        return count;
    }

    //922
    public int[] sortArrayByParityII(int[] A) {
        int[] result =new int[A.length];
        int odd = 1;
        int even = 0;
        for (int i =0; i< A.length; i++) {
            if (A[i] % 2 == 0 ){
                result[even] = A[i];
                even +=2;
            } else {
                result[odd] = A[i];
                odd +=2;
            }
        }
        return result;
    }
    public static void main(String[] args) {
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        map.put('a', 1);
        map.put('b', 1);
        System.out.println(map.get('a') == map.get('b'));
        map.put('c', 127);
        map.put('d', -128);
        System.out.println(map.get('c') == map.get('d'));
        System.out.println(map.get('c').equals(map.get('d')));
    }
}