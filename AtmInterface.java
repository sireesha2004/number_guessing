import java.util.*;
class AtmInterface
{
    public static void main(String args[]){
        int balance=10000,deposite,withdraw,num;
        System.out.println("---------------------------------------------------------");
         System.out.println("\nWelcome to ATM Machine\n");
         System.out.println("---------------------------------------------------------");
        Scanner sc = new Scanner(System.in);
        while(true)
        {
           
            System.out.println("\n1.Withdraw \n2.Deposite \n3.CheckBalance \n4.Transaction History \n5.Exit\n");
            System.out.println("\nEnter your choice:");
            num=sc.nextInt();
            switch(num)
            {
                case 1:
                    System.out.println("Enter the amount to withdraw: ");
                    withdraw=sc.nextInt();
                    if(withdraw<=balance){
                        balance-=withdraw;
                        System.out.println("Balance in Your account :  "+balance);
                        System.out.println("Please collect the money ");
                    }
                    else{
                        System.out.println("Sufficient amount is not available to withdraw!");
                    }
                    System.out.println(" ");
                    System.out.println("---------------------------------------------------------");
                    break;
                case 2:
                    System.out.println("Enter the amount to Deposite: ");
                    deposite=sc.nextInt();
                    System.out.println("amount deposited : "+deposite);
                    balance+=deposite;
                    System.out.println("The Amount in your account :  "+balance);
                    System.out.println("  ");
                    System.out.println("---------------------------------------------------------");
                    break;
                case 3:
                    System.out.println("Balance in Your account :  "+balance);
                    System.out.println("  ");
                    System.out.println("---------------------------------------------------------");
                    break;
                case 4:
                    System.out.println("These are transactions: \n ");
                    System.out.println("Withdrawn amount is: 2500\n");
                    System.out.println("Credited amount is: 5000\n ");
                    System.out.println("Balance checked\n ");
                    System.out.println("---------------------------------------------------------");
                    break;
                case 5:
                    System.out.println("--Please Take the card out--");
                    System.out.println("You are Exited\n ");
                    System.exit(0);
                    System.out.println("  ");
                    System.out.println("---------------------------------------------------------");
                    break;
                   
            }
        }
    }
}