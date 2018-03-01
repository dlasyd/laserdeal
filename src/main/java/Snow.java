import java.util.ArrayList;
import java.util.List;

public class Snow {
    public static void main(String[] args) {
        List<String> names = new ArrayList<>();

        int j =0;
        for (String name : names) {
            System.out.println("no index");
            j++;
        }

        for (int i = 0; i < names.size(); i++) {
            System.out.println("no instance");
            String name = names.get(i);
        }
    }
}
